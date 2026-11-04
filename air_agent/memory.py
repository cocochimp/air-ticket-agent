"""工作四：短期会话记忆（订单栈 + 滑动窗口摘要）、长期用户画像、Token 预算分配。"""
import json
import math
import threading
from . import data, seed
from . import settings as S
from .state import INTENT_LABEL

SYSTEM_PROMPT = ("你是航空公司退改签助手。只能依据检索到的规则和运价引擎返回的金额作答；"
                 "金额必须引用工具结果，不得自行计算或编造；拿不准时说明需要转人工。")
SUMMARY_MAX_CHARS = 600
ARCHIVE_MAX_TURNS = 50
PRIORITY = ["system", "order", "rules", "history", "profile"]


def tokens(text: str) -> int:
    return math.ceil(len(text) / 1.5)


_locks: dict[str, threading.Lock] = {}
_locks_guard = threading.Lock()


def session_lock(session_id: str) -> threading.Lock:
    """每个会话一把锁。图运行器必须在整轮对话（load_session -> save_session）外层持有它，
    否则同一会话的并发请求会互相覆盖记忆。"""
    with _locks_guard:
        return _locks.setdefault(session_id, threading.Lock())


def _empty():
    return {"order_stack": [], "progress": {}, "turns": [], "summary": "", "last_intent": None}


def load_session(session_id: str) -> dict:
    with data.connect() as c:
        r = c.execute("SELECT data FROM session_memory WHERE session_id=?", (session_id,)).fetchone()
    return {**_empty(), **json.loads(r["data"])} if r else _empty()


def save_session(session_id: str, user_id: str, mem: dict):
    with data.connect() as c:
        c.execute("INSERT OR REPLACE INTO session_memory VALUES(?,?,?,?)",
                  (session_id, user_id, json.dumps(mem, ensure_ascii=False), S.now().isoformat()))


def push_order(mem: dict, order_id: str):
    cap = S.get()["memory"]["order_stack"]
    mem["order_stack"] = ([order_id] + [x for x in mem["order_stack"] if x != order_id])[:cap]


def summarize(turns: list[dict], llm=None) -> str:
    if llm is not None:
        text = llm.summarize(turns)
        if text:
            return text
    return "；".join(f"{'用户' if t['role'] == 'user' else '助手'}：{t['text'][:40]}" for t in turns)


def add_turn(mem: dict, role: str, text: str, llm=None):
    mem["turns"].append({"role": role, "text": text})
    keep = S.get()["memory"]["recent_turns"] * 2
    if len(mem["turns"]) > keep:
        old, mem["turns"] = mem["turns"][:-keep], mem["turns"][-keep:]
        mem["archive"] = (mem.get("archive", []) + old)[-ARCHIVE_MAX_TURNS:]
        piece = summarize(old, llm)
        mem["summary"] = (f"{mem['summary']}；{piece}" if mem["summary"] else piece)[-SUMMARY_MAX_CHARS:]


def _bump(d: dict, key: str):
    d[key] = d.get(key, 0) + 1


def update_profile(user_id: str, order: dict, intent: str, key: str):
    if not order:
        return
    f = data.get_flight(order["flight_no"], order["date"])
    with data.transaction() as conn:          # 读-改-写在同一个写事务里，并发更新不丢计数
        prof = data.get_profile(user_id, conn)
        if not prof:
            return
        p = prof["data"]
        seen = p.setdefault("_seen", [])
        if key in seen:
            return
        seen.append(key)
        p["_seen"] = seen[-200:]
        if f:
            _bump(p.setdefault("routes", {}), f"{f['origin']}-{f['dest']}")
        _bump(p.setdefault("airlines", {}), order["airline"])
        _bump(p.setdefault("products", {}), order["fare_product"])
        if intent in ("CHANGE", "REFUND"):
            _bump(p.setdefault("prefer_action", {}), intent)
        data.save_profile(user_id, p, conn)


def clear_profile(user_id: str):
    data.save_profile(user_id, {"routes": {}, "airlines": {}, "products": {}, "prefer_action": {},
                                "price_sensitivity": "低"})


def profile_boost(user_id: str) -> dict:
    prof = data.get_profile(user_id)
    if not prof:
        return {"airlines": [], "products": []}
    p = prof["data"]
    return {"airlines": [k for k, v in p.get("airlines", {}).items() if v >= 3],
            "products": [k for k, v in p.get("products", {}).items() if v >= 3]}


def public_profile(prof: dict | None) -> dict | None:
    """给界面/上下文用的画像：去掉以下划线开头的内部字段（如 _seen 去重键）。"""
    if not prof:
        return prof
    return {**prof, "data": {k: v for k, v in prof["data"].items() if not k.startswith("_")}}


def profile_text(prof: dict | None) -> str:
    if not prof:
        return ""
    p = prof["data"]
    airline = lambda k: seed.AIRLINES.get(k, k)
    product = lambda k: seed.PRODUCTS[k][1] if k in seed.PRODUCTS else k
    intent = lambda k: INTENT_LABEL.get(k, k)

    def top(d, label=lambda k: k):
        return "、".join(label(k) for k, _ in sorted(d.items(), key=lambda x: -x[1])[:3]) or "无"
    return (f"{prof['name']}（{prof['tier']}）常走航线：{top(p.get('routes', {}))}；"
            f"偏好航司：{top(p.get('airlines', {}), airline)}；常买票种：{top(p.get('products', {}), product)}；"
            f"退改偏好：{top(p.get('prefer_action', {}), intent)}；价格敏感度：{p.get('price_sensitivity', '中')}")


def order_text(order: dict | None) -> str:
    if not order:
        return "未指定订单"
    f = data.get_flight(order["flight_no"], order["date"])
    return (f"订单{order['order_id']}：{seed.AIRLINES[order['airline']]} {order['flight_no']} "
            f"{f['origin_city']}→{f['dest_city']} {order['date']} {f['dep_time']} "
            f"{seed.PRODUCTS[order['fare_product']][1]} 票面价{order['fare']}元 航班状态{f['status']}")


def _history_text(mem: dict, full: bool) -> str:
    turns = (mem.get("archive", []) + mem["turns"]) if full else mem["turns"]
    lines = [f"{'用户' if t['role'] == 'user' else '助手'}：{t['text']}" for t in turns]
    if not full and mem["summary"]:
        lines.insert(0, f"早前摘要：{mem['summary']}")
    return "\n".join(lines)


def _fit_history(mem: dict, quota: int) -> str:
    """先保证最近几轮（超长时丢最旧的，保留最新），剩余额度再放摘要的末尾（最新部分）。"""
    cap = int(quota * 1.5)
    recent = "\n".join(f"{'用户' if t['role'] == 'user' else '助手'}：{t['text']}" for t in mem["turns"])
    recent = recent[-cap:] if cap > 0 else ""
    summary = f"早前摘要：{mem['summary']}" if mem["summary"] else ""
    room = cap - len(recent) - 1
    if not summary or room <= 0:
        return recent
    head = summary if len(summary) <= room else "…" + summary[-(room - 1):] if room > 1 else ""
    return f"{head}\n{recent}" if head and recent else (head or recent)


def _rules_text(rules: list[dict]) -> str:
    return "\n".join(f"[{r['id']}] {r['title']}：{r['content']}" for r in rules)


def build_context(order, rules, mem, profile) -> dict:
    cfg = S.get()["memory"]
    budget, share = cfg["token_budget"], cfg["budget_share"]
    k = S.get()["retrieval"]["k_inject"]
    pub = public_profile(profile)
    raw = {"system": SYSTEM_PROMPT, "order": order_text(order), "rules": _rules_text(rules[:k]),
           "history": _history_text(mem, full=False), "profile": profile_text(pub)}
    full = {**raw, "rules": _rules_text(rules), "history": _history_text(mem, full=True),
            "profile": json.dumps(pub["data"], ensure_ascii=False) if pub else ""}
    blocks, used, carry = {}, 0, 0
    for name in PRIORITY:              # 高优先级先拿；某块没用完的份额顺延（carry）给后面的块
        remaining = budget - used
        need = tokens(raw[name])
        if name in ("system", "order"):    # 这两块永不截断（settings 保证预算放得下）
            quota = max(need, min(remaining, int(budget * share[name]) + carry))
        elif name == PRIORITY[-1]:
            quota = remaining
        else:
            quota = min(remaining, int(budget * share[name]) + carry)
        text = raw[name]
        if need > quota:
            text = _fit_history(mem, quota) if name == "history" else text[:int(quota * 1.5)]
        t = tokens(text)
        blocks[name] = {"text": text, "tokens": t, "budget": quota, "truncated": text != raw[name]}
        used += t
        carry = max(0, quota - t)
    total = sum(b["tokens"] for b in blocks.values())
    full_total = sum(tokens(v) for v in full.values())
    return {"blocks": blocks, "total": total, "budget": budget, "full_total": full_total,
            "saved_pct": round(100 * (1 - total / full_total), 1) if full_total else 0.0}
