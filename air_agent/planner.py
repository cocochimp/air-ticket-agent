"""ReAct 决策：输出 {thought, action, args, source}。action 为工具名、finish、clarify 或 human。"""
import json
from . import data, query, seed
from .llm import LLMError
from .state import INTENT_LABEL, ParsedQuery, SubQuery

PLANS = {"CHANGE": ["tool_flight_status", "tool_rule_retrieve", "tool_recalc_fee"],
         "REFUND": ["tool_flight_status", "tool_rule_retrieve", "tool_recalc_fee"],
         "POLICY": ["tool_rule_retrieve"],
         "STATUS": ["tool_flight_status"]}
THOUGHTS = {"tool_flight_status": "先确认航班状态：取消或延误 2 小时以上属于非自愿退改，费用规则不同。",
            "tool_rule_retrieve": "检索适用的退改规则，确定收费档位和依据。",
            "tool_recalc_fee": "规则已确认，调用运价引擎计算金额（金额只能来自引擎）。",
            "finish": "所需信息已齐全，进入方案生成。"}


def current(state) -> tuple[ParsedQuery, SubQuery]:
    p = ParsedQuery.model_validate(state["parsed"])
    return p, p.sub_queries[state.get("sub_idx", 0)]


def build_args(tool, p: ParsedQuery, sub: SubQuery) -> dict:
    if tool == "tool_flight_status":
        o = data.get_order(p.order_id)
        return {"flight_no": o["flight_no"], "date": o["date"]}
    if tool == "tool_rule_retrieve":
        scen = query.scenario_for(p, sub)
        name = seed.AIRLINES.get(p.airline or "", "")
        prod = seed.PRODUCTS.get(p.fare_product or "", ("", ""))[1]
        args = {"query": f"{name}{prod}{query.KIND_LABEL.get(scen or '', '')} {sub.text}".strip()}
        for k in ("airline", "fare_product"):
            if getattr(p, k):
                args[k] = getattr(p, k)
        if scen:
            args["scenario"] = scen
        return args
    if tool == "tool_recalc_fee":
        args = {"order_id": p.order_id, "action": sub.intent}
        if sub.intent == "CHANGE":
            args["target_date"] = sub.target_date
        return args
    return {}


def rule_decide(state) -> dict:
    p, sub = current(state)
    for err in state.get("error_feedback") or []:
        if err.get("kind") == "SKIP_RULE":      # 防跳步打回：按计划表继续即可
            continue
        if err.get("field") == "target_date":
            return {"thought": f"日期「{sub.target_date}」无法换算成具体日期，需要向乘客确认。",
                    "action": "clarify", "args": {"field": "target_date"}, "source": "rule"}
        return {"thought": f"工具参数错误且无法自动修正：{err.get('message', err)}", "action": "human",
                "args": {}, "source": "rule"}
    done = {c["tool"] for c in state.get("calls", []) if c.get("ok")}
    for tool in PLANS[sub.intent]:
        if tool not in done:
            return {"thought": THOUGHTS[tool], "action": tool, "args": build_args(tool, p, sub), "source": "rule"}
    return {"thought": THOUGHTS["finish"], "action": "finish", "args": {}, "source": "rule"}


def llm_messages(state) -> list[dict]:
    p, sub = current(state)
    ctx = state.get("context", {}).get("blocks", {})
    sys = ctx.get("system", {}).get("text", "")
    obs = [{"tool": c["tool"], "args": c.get("args"), "ok": c.get("ok"),
            "result": c.get("output") if c.get("ok") else c.get("errors") or c.get("error")}
           for c in state.get("calls", [])]
    user = (f"{ctx.get('order', {}).get('text', '')}\n用户画像：{ctx.get('profile', {}).get('text', '')}\n"
            f"会话：{ctx.get('history', {}).get('text', '')}\n"
            f"当前子诉求（{INTENT_LABEL[sub.intent]}）：{sub.text}；目标日期：{sub.target_date}\n"
            f"已执行：{json.dumps(obs, ensure_ascii=False)}\n"
            f"上次参数错误：{json.dumps(state.get('error_feedback') or [], ensure_ascii=False)}\n"
            f"今天是 {query.today().isoformat()}。先说一句思考，再调用一个工具；信息齐全时调用 finish。"
            f"算钱前必须先检索规则；日期必须是 YYYY-MM-DD。")
    return [{"role": "system", "content": sys}, {"role": "user", "content": user}]


def decide(state, llm=None) -> dict:
    if llm is None:
        return rule_decide(state)
    try:
        a = llm.react(llm_messages(state))
        return {**a, "source": "llm"}
    except LLMError as e:
        a = rule_decide(state)
        return {**a, "source": "fallback", "fallback_reason": str(e)}
