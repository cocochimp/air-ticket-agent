"""金标准评测：构建用例、校验用例、批量运行、计算指标。

CLI（在独立沙箱里运行，不影响演示数据）：
  uv run python -m air_agent.evaluation build              # 生成 eval/gold_cases.json
  uv run python -m air_agent.evaluation validate           # 校验用例，输出 JSON
  uv run python -m air_agent.evaluation run --mode offline # 运行并写入 var/eval_runs/<run_id>.json
"""
import argparse
import json
import shutil
import statistics
import sys
import time
import uuid
from collections import Counter, defaultdict
from datetime import date, datetime, timedelta
from typing import Literal
from pydantic import BaseModel, Field, ValidationError
from . import data, engine, graph, tools
from . import settings as S

EVAL_NOW = "2026-09-26T10:00"
CASES_PATH = S.ROOT / "eval" / "gold_cases.json"
RUNS_DIR = S.ROOT / "var" / "eval_runs"
REACT = ["tool_flight_status", "tool_rule_retrieve", "tool_recalc_fee"]
FORBID = ["quote_mismatch", "unsupported_claim", "skip_rule"]


class Expect(BaseModel):
    intents: list[Literal["CHANGE", "REFUND", "POLICY", "STATUS"]]
    tools: list[str]
    fees: dict[str, int | None] = Field(default_factory=dict)
    rule_ids: list[str] = Field(default_factory=list)
    outcome: Literal["ANSWER", "HUMAN", "CLARIFY"]
    schema_errors: list[str] = Field(default_factory=list)


class Turn(BaseModel):
    message: str
    order: str | None = None


class Case(BaseModel):
    id: str = Field(pattern=r"^[\w-]{1,64}$")
    title: str
    tags: list[str] = Field(default_factory=list)
    user_id: str
    airline: str | None = None
    fare_product: str | None = None
    order: str | None = None
    question: str = Field(min_length=1, max_length=1000)
    history: list[Turn] = Field(default_factory=list)
    clarify_answers: list[str] = Field(default_factory=list)
    expect: Expect
    forbid: list[str] = Field(default_factory=lambda: list(FORBID))


# ---------------- 订单引用 ----------------
def order_id(ref: str | None) -> str | None:
    if not ref:
        return None
    if ref.startswith("#"):
        return f"ORD{datetime.fromisoformat(EVAL_NOW):%Y%m%d}{int(ref[1:]):03d}"
    return ref


def _md(d: str) -> str:
    x = date.fromisoformat(d)
    return f"{x.month}月{x.day}号"


def _fee_expect(oid, intent, target=None):
    r = engine.calc(oid, intent, target_date=target)
    if not r["allowed"]:
        return None, r["rule_id"]
    return (r["total"] if intent == "CHANGE" else r["refund_amount"]), r["rule_id"]


# ---------------- 构建 ----------------
def build_cases() -> list[dict]:
    """按 12 张演示订单 × 4 类问法 + 政策 / 追问 / 指代 / 转人工 生成初始用例。"""
    data.init(force=True)
    cases = []
    full = ["tool_flight_status", "tool_rule_retrieve", "tool_recalc_fee"]
    for n in range(1, 13):
        ref = f"#{n}"
        o = data.get_order(order_id(ref))
        f = data.get_flight(o["flight_no"], o["date"])
        future = engine.departure(f) > S.now()
        target = (max(date.fromisoformat(o["date"]), S.now().date()) + timedelta(days=1)).isoformat()
        base_tags = []
        if engine.involuntary_reason(f):
            base_tags.append("非自愿")
        if not future:
            base_tags.append("误机")
        chg, chg_rule = _fee_expect(o["order_id"], "CHANGE", target)
        rfd, rfd_rule = _fee_expect(o["order_id"], "REFUND")
        common = {"user_id": o["user_id"], "airline": o["airline"], "fare_product": o["fare_product"], "order": ref}
        variants = [
            ("chg", f"帮我把这张票改到{_md(target)}", ["CHANGE"], full, {"CHANGE": chg}, [chg_rule], ["单诉求"]),
            ("rfd", "这张票退掉能退多少钱", ["REFUND"], full, {"REFUND": rfd}, [rfd_rule], ["单诉求"]),
            ("both", f"改到{_md(target)}要多少钱，顺便看看退票多少", ["CHANGE", "REFUND"], full + full,
             {"CHANGE": chg, "REFUND": rfd}, sorted({chg_rule, rfd_rule}), ["多诉求"]),
            ("status", "这个航班现在什么情况", ["STATUS"], ["tool_flight_status"], {}, [], ["航班状态"]),
        ]
        for key, q, intents, tl, fees, rules, tags in variants:
            extra = ["不可办理"] if any(v is None for v in fees.values()) else []
            cases.append({"id": f"o{n:02d}-{key}", "title": f"订单#{n} {o['airline']} {o['fare_product']} {q}",
                          "tags": tags + base_tags + extra, **common, "question": q, "history": [],
                          "clarify_answers": [],
                          "expect": {"intents": intents, "tools": tl, "fees": fees, "rule_ids": rules,
                                     "outcome": "ANSWER", "schema_errors": []}, "forbid": list(FORBID)})
    # 政策咨询用例刻意不带「退票/改签」等办理关键词：这些用户都有已出票订单，
    # 带办理关键词时 query 模块会先追问订单（这是期望行为，由 clarify-order 覆盖）
    policy = [("MU", "SPECIAL", "东方云航特价票的退订规定是什么", "MU-SPECIAL-RFD"),
              ("CA", "F", "国际星航头等舱更改航班有什么规定", "CA-F-CHG"),
              ("CZ", "Y_DISCOUNT", "南方海航折扣票退订手续费的规定", "CZ-Y_DISCOUNT-RFD"),
              ("HU", "GROUP", "海岛航空团队票更改行程的规定", "HU-GROUP-CHG"),
              ("MU", "C", "东方云航公务舱退订政策", "MU-C-RFD"),
              ("CA", "Y", "国际星航经济舱全价票更改航班的规则", "CA-Y-CHG"),
              ("CZ", "F", "南方海航头等舱退订规则", "CZ-F-RFD"),
              ("HU", "Y_DISCOUNT", "海岛航空打折票更改航班政策", "HU-Y_DISCOUNT-CHG")]
    for i, (al, prod, q, rid) in enumerate(policy, 1):
        cases.append({"id": f"policy-{i:02d}", "title": q, "tags": ["政策咨询"], "user_id": "u_wang",
                      "airline": al, "fare_product": prod, "order": None, "question": q, "history": [],
                      "clarify_answers": [],
                      "expect": {"intents": ["POLICY"], "tools": ["tool_rule_retrieve"], "fees": {}, "rule_ids": [rid],
                                 "outcome": "ANSWER", "schema_errors": []}, "forbid": list(FORBID)})
    o1, o3 = data.get_order(order_id("#1")), data.get_order(order_id("#3"))
    r3, r3_rule = _fee_expect(o3["order_id"], "REFUND")
    c1, c1_rule = _fee_expect(o1["order_id"], "CHANGE", "2026-09-29")
    c1b, _ = _fee_expect(o1["order_id"], "CHANGE", "2026-10-02")
    r1, r1_rule = _fee_expect(o1["order_id"], "REFUND")
    o2 = data.get_order(order_id("#2"))
    r2, r2_rule = _fee_expect(o2["order_id"], "REFUND")
    c3, c3_rule = _fee_expect(o3["order_id"], "CHANGE", "2026-10-08")
    mk = lambda **kw: {"tags": [], "user_id": "u_zhang", "airline": None, "fare_product": None, "order": None,
                       "history": [], "clarify_answers": [], "forbid": list(FORBID), **kw}
    cases += [
        mk(id="clarify-order", title="没指定订单，追问后选择订单#3", tags=["追问"], airline="CA", fare_product="C",
           question="我要退票", clarify_answers=["#3"],
           expect={"intents": ["REFUND"], "tools": full, "fees": {"REFUND": r3}, "rule_ids": [r3_rule],
                   "outcome": "ANSWER", "schema_errors": []}),
        mk(id="clarify-date", title="改签没说日期，追问后选 9月29日", tags=["追问"], airline="MU",
           fare_product="Y_DISCOUNT", order="#1", question="我想改签", clarify_answers=["2026-09-29"],
           expect={"intents": ["CHANGE"], "tools": full, "fees": {"CHANGE": c1}, "rule_ids": [c1_rule],
                   "outcome": "ANSWER", "schema_errors": []}),
        mk(id="clarify-vague-date", title="模糊日期触发 Schema 校验，再追问", tags=["追问", "Schema"], airline="MU",
           fare_product="Y_DISCOUNT", order="#1", question="改到下个月初", clarify_answers=["2026-10-02"],
           expect={"intents": ["CHANGE"], "tools": full, "fees": {"CHANGE": c1b}, "rule_ids": [c1_rule],
                   "outcome": "ANSWER", "schema_errors": ["TYPE"]}),
        mk(id="coref-return", title="上一轮提到去程，这一轮问回程", tags=["指代"], airline="MU", fare_product="Y_DISCOUNT",
           history=[{"message": "帮我把杭州飞北京那张改到9月29号", "order": None}], question="回程那张能退吗",
           expect={"intents": ["REFUND"], "tools": full, "fees": {"REFUND": r2}, "rule_ids": [r2_rule],
                   "outcome": "ANSWER", "schema_errors": []}),
        mk(id="coref-ellipsis", title="省略追问：那退票呢", tags=["指代"], airline="MU", fare_product="Y_DISCOUNT",
           history=[{"message": "改到9月29号", "order": "#1"}], question="那退票呢",
           expect={"intents": ["REFUND"], "tools": full, "fees": {"REFUND": r1}, "rule_ids": [r1_rule],
                   "outcome": "ANSWER", "schema_errors": []}),
        mk(id="coref-stack", title="订单栈：上一轮问#3，这一轮只说日期", tags=["指代"], airline="CA", fare_product="C",
           history=[{"message": "退票多少钱", "order": "#3"}], question="那改到10月8号呢",
           expect={"intents": ["CHANGE"], "tools": full, "fees": {"CHANGE": c3}, "rule_ids": [c3_rule],
                   "outcome": "ANSWER", "schema_errors": []}),
    ]
    for i, q in enumerate(["宠物能带上飞机吗", "会员积分怎么兑换里程", "机上餐食可以选素食吗"], 1):
        # 意图是猜的（置信度低），query 模块先追问意图；回答「政策咨询」后检索不到依据 → 转人工
        cases.append(mk(id=f"human-{i:02d}", title=q, tags=["转人工", "追问"], user_id="u_wang", question=q,
                        clarify_answers=["POLICY"],
                        expect={"intents": ["POLICY"], "tools": ["tool_rule_retrieve"], "fees": {}, "rule_ids": [],
                                "outcome": "HUMAN", "schema_errors": []}))
    return cases


# ---------------- 校验 ----------------
def validate_cases(cases: list[dict]) -> dict[str, list[dict]]:
    """返回 {case_id: [{field, message}]}；没有错误的用例不出现。"""
    data.init(force=True)
    out: dict[str, list[dict]] = defaultdict(list)
    seen = Counter(c.get("id") for c in cases)
    rule_ids = {r["id"] for r in data.list_rules()}
    users = {p["user_id"] for p in data.list_profiles()}
    for raw in cases:
        cid = raw.get("id") or "?"
        try:
            c = Case.model_validate(raw)
        except ValidationError as e:
            for err in e.errors():
                out[cid].append({"field": ".".join(str(x) for x in err["loc"]), "message": err["msg"]})
            continue
        if seen[c.id] > 1:
            out[cid].append({"field": "id", "message": "用例 ID 重复"})
        if c.user_id not in users:
            out[cid].append({"field": "user_id", "message": f"用户不存在：{c.user_id}"})
        oid = order_id(c.order)
        if c.order and not data.get_order(oid):
            out[cid].append({"field": "order", "message": f"订单不存在：{c.order}"})
        for i, h in enumerate(c.history):
            if h.order and not data.get_order(order_id(h.order)):
                out[cid].append({"field": f"history.{i}.order", "message": f"订单不存在：{h.order}"})
        bad_tools = [t for t in c.expect.tools if t not in REACT]
        if bad_tools:
            out[cid].append({"field": "expect.tools", "message": f"非法工具：{bad_tools}"})
        bad_rules = [r for r in c.expect.rule_ids if r not in rule_ids]
        if bad_rules:
            out[cid].append({"field": "expect.rule_ids", "message": f"规则不存在：{bad_rules}"})
        fee_order = oid or order_id(next((a for a in c.clarify_answers if a.startswith("#")), None)) \
            or _last_history_order(c)
        for intent, val in c.expect.fees.items():
            if intent not in ("CHANGE", "REFUND"):
                out[cid].append({"field": f"expect.fees.{intent}", "message": "只能是 CHANGE 或 REFUND"})
                continue
            if not fee_order or not data.get_order(fee_order):
                out[cid].append({"field": f"expect.fees.{intent}", "message": "无法确定订单，不能校验金额"})
                continue
            target = _case_target(c) if intent == "CHANGE" else None
            try:
                got, _ = _fee_expect(fee_order, intent, target)
            except engine.EngineError as e:
                out[cid].append({"field": f"expect.fees.{intent}", "message": f"引擎报错：{e}"})
                continue
            if got != val:
                out[cid].append({"field": f"expect.fees.{intent}", "message": f"期望金额与引擎不符，引擎值为 {got}"})
    return dict(out)


def _last_history_order(c: Case):
    """指代类用例：订单来自历史或代词，取能推出的订单（与 build_cases 的写法一致）。"""
    from . import query
    ctx = {"user_id": c.user_id, "selected_order_id": None, "order_stack": [], "last_intent": None}
    for h in c.history:
        ctx["selected_order_id"] = order_id(h.order)
        oid = query.resolve_order(h.message, ctx)
        if oid:
            ctx["order_stack"] = [oid] + ctx["order_stack"]
        ctx["selected_order_id"] = None
    return query.resolve_order(c.question, ctx)


def _case_target(c: Case):
    from . import query
    for a in c.clarify_answers:
        if not a.startswith("#"):
            return a
    t = query.find_date(c.question)
    return t if t and t[:2] == "20" else None


# ---------------- 运行 ----------------
def run_case(case: dict, mode: str, run_id: str, client=None) -> dict:
    c = Case.model_validate(case)
    data.init(force=True)
    sid = f"eval-{run_id}-{c.id}"
    for h in c.history:
        graph.run_turn(sid, c.user_id, h.message, order_id=order_id(h.order), mode=mode, client=client)
    t0 = time.perf_counter()
    res = graph.run_turn(sid, c.user_id, c.question, order_id=order_id(c.order), mode=mode, client=client)
    answers = list(c.clarify_answers)
    trace = list(res["trace"])
    while res["interrupt"] and res["interrupt"]["kind"] == "clarify" and answers:
        a = answers.pop(0)
        res = graph.resume(sid, {"value": order_id(a) if a.startswith("#") else a}, client=client)
        trace += res["trace"]
    ms = round((time.perf_counter() - t0) * 1000, 1)
    outcome = "CLARIFY" if res["interrupt"] and res["interrupt"]["kind"] == "clarify" else res["outcome"]
    results = res["results"] or []
    final = res["final"] or {}
    tools_seq = [x["tool"] for r in results for x in r["calls"] if x["ok"] and x["tool"] in REACT]
    schema_errs = [e["output"][0]["kind"] for e in trace if e["node"] == "schema_validate" and e["status"] == "error"]
    blocked = any(e["node"] == "checkpoint_gate" and e["status"] == "blocked" for e in trace)
    retrieved = {x["id"] for r in results for x in r["rules"]}
    fees = {}
    for opt, r in zip(final.get("options", []), results):
        if r["intent"] in ("CHANGE", "REFUND"):
            fees[r["intent"]] = opt["total"] if opt.get("allowed") else None
    v = res["verify"] or {}
    reasons = []
    tool_ok = tools_seq == c.expect.tools and all(k in c.expect.schema_errors for k in schema_errs)
    if tools_seq != c.expect.tools:
        reasons.append("wrong_tools")
    reasons += [k for k in schema_errs if k not in c.expect.schema_errors]
    hit = set(c.expect.rule_ids) <= retrieved if c.expect.rule_ids else None
    if hit is False:
        reasons.append("rule_miss")
    if outcome != c.expect.outcome:
        reasons.append("wrong_outcome")
    if outcome == "ANSWER" and [r["intent"] for r in results] != c.expect.intents:
        reasons.append("wrong_intent")
    if outcome == "ANSWER" and any(fees.get(k) != val for k, val in c.expect.fees.items()):
        reasons.append("fee_mismatch")
    if "skip_rule" in c.forbid and blocked:
        reasons.append("skip_rule")
    if outcome == "ANSWER" and "quote_mismatch" in c.forbid and v.get("quote_mismatch"):
        reasons.append("quote_mismatch")
    if outcome == "ANSWER" and "unsupported_claim" in c.forbid and v.get("unsupported_claims"):
        reasons.append("unsupported_claim")
    passed = not [r for r in reasons if r not in ("wrong_tools", "rule_miss", *c.expect.schema_errors)] \
        and outcome == c.expect.outcome
    tokens = (res.get("context") or {}).get("total", 0)
    return {"case_id": c.id, "title": c.title, "tags": c.tags,
            "group": f"{c.airline or '通用'}·{c.fare_product or '通用'}", "passed": passed, "tool_ok": tool_ok,
            "policy_hit": hit, "reasons": reasons, "ms": ms, "tokens": tokens, "expected": c.expect.model_dump(),
            "actual": {"tools": tools_seq, "intents": [r["intent"] for r in results], "fees": fees,
                       "outcome": outcome, "rule_ids": sorted(retrieved), "text": final.get("text", ""),
                       "schema_errors": schema_errs},
            "trace": trace}


def summarize(results: list[dict]) -> dict:
    n = len(results) or 1
    lat = sorted(r["ms"] for r in results) or [0]
    hits = [r["policy_hit"] for r in results if r["policy_hit"] is not None]
    groups = defaultdict(lambda: {"n": 0, "passed": 0})
    for r in results:
        groups[r["group"]]["n"] += 1
        groups[r["group"]]["passed"] += r["passed"]
    return {"n": len(results),
            "tool_acc": round(sum(r["tool_ok"] for r in results) / n, 4),
            "policy_hit": round(sum(hits) / len(hits), 4) if hits else None,
            "correctness": round(sum(r["passed"] for r in results) / n, 4),
            "human_rate": round(sum(r["actual"]["outcome"] == "HUMAN" for r in results) / n, 4),
            "latency_avg_ms": round(statistics.mean(lat), 1),
            "latency_p95_ms": lat[min(len(lat) - 1, int(len(lat) * 0.95))],
            "tokens_avg": round(statistics.mean([r["tokens"] for r in results]), 1) if results else 0,
            "failure_reasons": dict(Counter(x for r in results for x in r["reasons"])),
            "by_group": {k: {**v, "rate": round(v["passed"] / v["n"], 4)} for k, v in sorted(groups.items())}}


def run_cases(cases: list[dict], mode: str = "offline", run_id: str | None = None, on_progress=None, client=None):
    run_id = run_id or f"{S.now():%Y%m%d%H%M}-{uuid.uuid4().hex[:6]}"
    results = []
    for i, c in enumerate(cases, 1):
        try:
            r = run_case(c, mode, run_id, client)
        except Exception as e:      # 单条用例出错不影响整批，记为失败
            r = {"case_id": c.get("id"), "title": c.get("title", ""), "tags": c.get("tags", []), "group": "错误",
                 "passed": False, "tool_ok": False, "policy_hit": None, "reasons": [f"error:{type(e).__name__}"],
                 "ms": 0, "tokens": 0, "expected": c.get("expect"), "actual": {"outcome": "ERROR", "text": str(e)},
                 "trace": []}
        results.append(r)
        if on_progress:
            on_progress({"type": "progress", "done": i, "total": len(cases), "case_id": r["case_id"],
                         "passed": r["passed"]})
    return {"id": run_id, "created_at": datetime.now().isoformat(timespec="seconds"), "mode": mode,
            "config": S.get(), "summary": summarize(results), "results": results}


# ---------------- CLI ----------------
def _sandbox():
    sandbox = S.ROOT / "var" / "eval_sandbox"
    shutil.rmtree(sandbox, ignore_errors=True)
    S.VAR_DIR = sandbox
    S.set_now(datetime.fromisoformat(EVAL_NOW))
    graph.reset_graph()


def _emit(obj):
    sys.stdout.write(json.dumps(obj, ensure_ascii=False) + "\n")
    sys.stdout.flush()


def main(argv=None):
    ap = argparse.ArgumentParser(prog="air_agent.evaluation")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("build")
    v = sub.add_parser("validate")
    v.add_argument("--cases", default=str(CASES_PATH))
    r = sub.add_parser("run")
    r.add_argument("--cases", default=str(CASES_PATH))
    r.add_argument("--mode", default="offline", choices=["offline", "deepseek"])
    r.add_argument("--ids", default="")
    r.add_argument("--run-id", default=None)
    args = ap.parse_args(argv)
    runs_dir = RUNS_DIR
    _sandbox()
    if args.cmd == "build":
        CASES_PATH.parent.mkdir(parents=True, exist_ok=True)
        cases = build_cases()
        CASES_PATH.write_text(json.dumps(cases, ensure_ascii=False, indent=2), "utf-8")
        _emit({"type": "built", "count": len(cases), "path": str(CASES_PATH)})
        return
    cases = json.loads(open(args.cases, encoding="utf-8").read())
    if args.cmd == "validate":
        _emit({"type": "validated", "errors": validate_cases(cases)})
        return
    if args.ids:
        keep = set(args.ids.split(","))
        cases = [c for c in cases if c["id"] in keep]
    run = run_cases(cases, args.mode, args.run_id, on_progress=_emit)
    runs_dir.mkdir(parents=True, exist_ok=True)
    (runs_dir / f"{run['id']}.json").write_text(json.dumps(run, ensure_ascii=False), "utf-8")
    _emit({"type": "done", "run_id": run["id"], "summary": run["summary"]})


if __name__ == "__main__":
    main()
