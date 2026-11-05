"""LangGraph 状态图：工作一的编排，串起 Query 理解、检索、记忆、生成与校验。

阅读顺序：build_graph() 看节点和边 → 各节点函数 → run_turn()/resume() 看如何驱动与流式输出。
interrupt 注意：节点在 resume 后会从头重新执行，所以 interrupt() 之前不能有副作用。
"""
import sqlite3
import time
import uuid
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt
from . import data, llm as llm_mod, memory, planner, query, retrieval, tools
from . import scheme as scheme_mod
from . import settings as S
from . import verify as verify_mod
from .state import AgentState, ParsedQuery

_clients: dict = {}          # session_id -> LLM 客户端（不写进检查点）
_graph = None


def _llm(state):
    return _clients.get(state["session_id"])


def ev(node, t0, inp=None, out=None, note="", status="ok"):
    return {"node": node, "status": status, "ms": round((time.perf_counter() - t0) * 1000, 1),
            "at": S.now().isoformat(), "input": inp, "output": out, "note": note}


def _qctx(state):
    mem, cl = state.get("memory") or {}, state.get("clarified") or {}
    return {"user_id": state["user_id"], "selected_order_id": cl.get("order_id") or state.get("order_id"),
            "order_stack": mem.get("order_stack", []), "last_intent": mem.get("last_intent"),
            "forced_date": cl.get("target_date"), "forced_intent": cl.get("intent")}


# ---------------- 节点 ----------------
def user_input(state):
    t0 = time.perf_counter()
    return {"sub_idx": 0, "calls": [], "results": [], "rounds": 0, "tool_retries": 0, "gate_blocks": 0,
            "retries": 0, "feedback": [], "error_feedback": [], "clarified": {}, "clarify_req": None,
            "clarify_count": 0, "stepped_back": False, "outcome": "", "final": {}, "scheme": {}, "verify": {},
            "confirm": None,
            "trace": [ev("user_input", t0, {"question": state["question"], "order_id": state.get("order_id"),
                                            "mode": state.get("mode")}, None, "接收乘客诉求")]}


def memory_node(state):
    t0 = time.perf_counter()
    mem = memory.load_session(state["session_id"])
    view = {"order_stack": mem["order_stack"], "summary": mem["summary"], "turns": len(mem["turns"]),
            "last_intent": mem["last_intent"], "boost": memory.profile_boost(state["user_id"])}
    return {"memory": mem, "trace": [ev("memory_node", t0, None, view, "加载短期记忆与用户画像")]}


def order_align(state):
    t0 = time.perf_counter()
    oid = query.resolve_order(state["question"], _qctx(state))
    out = None
    if oid:
        o = data.get_order(oid)
        out = {"order_id": oid, "airline": o["airline"], "cabin": o["cabin"], "fare_product": o["fare_product"],
               "label": query.order_label(o)}
    return {"trace": [ev("order_align", t0, {"selected": state.get("order_id")}, out,
                         "订单字段对齐" if oid else "未能确定订单", "ok" if oid else "warn")]}


def query_understand(state):
    t0 = time.perf_counter()
    p = query.understand(state["question"], _qctx(state), _llm(state))
    order = data.get_order(p.order_id) if p.order_id else None
    ctx = memory.build_context(order, [], state["memory"], data.get_profile(state["user_id"]))
    note = f"{len(p.sub_queries)} 个子诉求，置信度 {p.confidence}" + ("，需要追问" if p.clarify else "")
    return {"parsed": p.model_dump(), "context": ctx,
            "trace": [ev("query_understand", t0, {"question": state["question"]}, p.model_dump(), note)]}


def clarify(state):
    t0 = time.perf_counter()
    req = state.get("clarify_req") or ParsedQuery.model_validate(state["parsed"]).clarify
    if state.get("clarify_count", 0) >= 3:
        return {"verify": {"errors": ["多次追问仍无法确定诉求"]}, "outcome": "HUMAN",
                "trace": [ev("clarify", t0, req, None, "追问次数已达上限", "error")]}
    answer = interrupt({"kind": "clarify", **req})
    value = answer.get("value") if isinstance(answer, dict) else answer
    return {"clarified": {**(state.get("clarified") or {}), req["field"]: value}, "clarify_req": None,
            "clarify_count": state.get("clarify_count", 0) + 1, "calls": [], "results": [], "sub_idx": 0,
            "rounds": 0, "tool_retries": 0, "gate_blocks": 0, "error_feedback": [], "stepped_back": False,
            "trace": [ev("clarify", t0, req, {"field": req["field"], "value": value}, "已获得乘客补充信息")]}


def agent_node(state):
    t0 = time.perf_counter()
    _, sub = planner.current(state)
    rounds = state.get("rounds", 0) + 1
    if rounds > S.get()["planner"]["max_rounds"]:
        a = {"thought": "已达到 ReAct 轮数上限，转人工", "action": "human", "args": {}, "source": "guard"}
    else:
        a = planner.decide(state, _llm(state))
    upd = {"rounds": rounds, "next_action": a,
           "trace": [ev("agent_node", t0, {"sub_query": sub.text, "intent": sub.intent, "round": rounds}, a,
                        f"第 {rounds} 轮 → {a['action']}")]}
    if a["action"] == "clarify":
        upd["clarify_req"] = {"field": "target_date", "options": query.date_options(),
                              "question": f"「{sub.target_date}」无法确定具体日期，想改签到哪一天？"}
    if a["action"] == "human":
        upd["verify"] = {"errors": [a["thought"]]}
    return upd


def schema_validate(state):
    t0 = time.perf_counter()
    a = state["next_action"]
    model, errs = tools.validate(a["action"], a["args"])
    if errs:
        call = {"tool": a["action"], "args": a["args"], "ok": False, "kind": errs[0]["kind"], "errors": errs}
        return {"tool_retries": state.get("tool_retries", 0) + 1, "error_feedback": errs,
                "calls": state["calls"] + [call],
                "trace": [ev("schema_validate", t0, {"tool": a["action"], "args": a["args"]}, errs,
                             f"{errs[0]['kind']}：{errs[0]['message']}", "error")]}
    clean = model.model_dump(exclude_none=True)
    return {"error_feedback": [], "next_action": {**a, "args": clean},
            "trace": [ev("schema_validate", t0, {"tool": a["action"], "args": a["args"]}, clean, "参数校验通过")]}


def checkpoint_gate(state):
    t0 = time.perf_counter()
    a = state["next_action"]
    done = {c["tool"] for c in state["calls"] if c.get("ok")}
    if a["action"] == "tool_recalc_fee" and "tool_rule_retrieve" not in done:
        fb = [{"field": "plan", "kind": "SKIP_RULE", "message": "计算金额前必须先调用 tool_rule_retrieve 检索规则"}]
        return {"gate_blocks": state.get("gate_blocks", 0) + 1, "error_feedback": fb, "gate_reason": fb[0]["message"],
                "trace": [ev("checkpoint_gate", t0, {"tool": a["action"], "done": sorted(done)}, None,
                             fb[0]["message"], "blocked")]}
    return {"gate_reason": "",
            "trace": [ev("checkpoint_gate", t0, {"tool": a["action"], "done": sorted(done)}, None, "必经节点检查通过")]}


def tool_exec(state):
    t0 = time.perf_counter()
    a = state["next_action"]
    r = tools.run_tool(a["action"], a["args"], memory.profile_boost(state["user_id"]))
    call = {"tool": a["action"], "args": a["args"], **r}
    upd = {"calls": state["calls"] + [call], "error_feedback": []}
    if r["ok"]:
        upd["tool_retries"] = 0
        note = {"tool_rule_retrieve": lambda o: f"命中 {len(o['rules'])} 条规则（{o['backend']}）",
                "tool_recalc_fee": lambda o: f"引擎计算完成：{o['rule_id']}",
                "tool_flight_status": lambda o: f"航班状态 {o['status']}"}[a["action"]](r["output"])
        upd["trace"] = [ev("tool_exec", t0, {"tool": a["action"], "args": a["args"]}, r["output"], note)]
    else:
        upd["trace"] = [ev("tool_exec", t0, {"tool": a["action"], "args": a["args"]}, r,
                           r.get("error", {}).get("message", "工具执行失败"), "error")]
    return upd


def step_back(state):
    t0 = time.perf_counter()
    p, sub = planner.current(state)
    last = state["calls"][-1]
    filters = {k: last["args"][k] for k in ("airline", "cabin", "fare_product", "scenario") if last["args"].get(k)}
    boost = memory.profile_boost(state["user_id"])
    q = query.step_back_query(p, sub)
    res, used = retrieval.search(q, filters, boost), "step_back"
    if not res["rules"]:
        q = query.hyde_text(p, sub)
        res, used = retrieval.search(q, filters, boost), "hyde"
    old = last["output"]["rules"]
    merged = old + [x for x in res["rules"] if x["id"] not in {o["id"] for o in old}]
    merged = merged[:S.get()["retrieval"]["k_inject"]]
    calls = state["calls"][:-1] + [{**last, "output": {**last["output"], "rules": merged},
                                    "rewrite": {"type": used, "query": q}}]
    return {"calls": calls, "stepped_back": True,
            "trace": [ev("step_back", t0, {"query": q, "type": used}, [x["id"] for x in merged],
                         f"{'Step-back 泛化' if used == 'step_back' else 'HyDE 扩写'}后共 {len(merged)} 条")]}


def sub_done(state):
    t0 = time.perf_counter()
    _, sub = planner.current(state)
    get = lambda tool: next((c for c in reversed(state["calls"]) if c["tool"] == tool and c.get("ok")), None)
    err = next((c for c in state["calls"] if c.get("kind") == "ENGINE"), None)
    rr, fs, fee = get("tool_rule_retrieve"), get("tool_flight_status"), get("tool_recalc_fee")
    result = {"intent": sub.intent, "text": sub.text, "target_date": sub.target_date,
              "status": fs["output"] if fs else None,
              "rules": rr["output"]["rules"] if rr else [], "funnel": rr["output"]["funnel"] if rr else [],
              "retrieval": {k: rr["output"][k] for k in ("backend", "degraded")} if rr else None,
              "rewrite": rr.get("rewrite") if rr else None,
              "fee": fee["output"] if fee else None, "error": err["error"]["message"] if err else None,
              "calls": [{k: c.get(k) for k in ("tool", "args", "ok", "kind")} for c in state["calls"]]}
    result = scheme_mod.ensure_fee_rule(result)     # verify 要求引擎依据规则被检索到并被引用
    return {"results": state["results"] + [result], "sub_idx": state["sub_idx"] + 1, "calls": [], "rounds": 0,
            "tool_retries": 0, "gate_blocks": 0, "error_feedback": [], "stepped_back": False,
            "trace": [ev("sub_done", t0, {"sub_query": sub.text}, {"rules": [x["id"] for x in result["rules"]],
                                                                  "fee": result["fee"], "error": result["error"]},
                         f"子诉求完成：{sub.intent}")]}


def tool_scheme_gen(state):
    t0 = time.perf_counter()
    p = ParsedQuery.model_validate(state["parsed"])
    order = data.get_order(p.order_id) if p.order_id else None
    rules = [x for r in state["results"] for x in r["rules"]]
    ctx = memory.build_context(order, rules, state["memory"], data.get_profile(state["user_id"]))
    out = scheme_mod.generate(state["results"], state.get("feedback"), _llm(state), ctx)
    return {"scheme": out.model_dump(), "context": ctx,
            "trace": [ev("tool_scheme_gen", t0, {"feedback": state.get("feedback")}, out.model_dump(),
                         f"上下文 {ctx['total']}/{ctx['budget']} tokens，较全量节省 {ctx['saved_pct']}%")]}


def verify_node(state):
    t0 = time.perf_counter()
    c = _llm(state)
    v = verify_mod.check(state["scheme"], state["results"], c.fact_check if c else None)
    return {"verify": v, "trace": [ev("verify_node", t0, None, v, f"置信度 {v['confidence']}",
                                      "ok" if v["passed"] else "error")]}


def retry_node(state):
    t0 = time.perf_counter()
    errs = state["verify"]["errors"]
    return {"retries": state.get("retries", 0) + 1, "feedback": errs,
            "trace": [ev("retry_node", t0, None, errs, f"第 {state.get('retries', 0) + 1} 次自我修正")]}


def human_review(state):
    t0 = time.perf_counter()
    reasons = ((state.get("verify") or {}).get("errors")
               or [e.get("message", "") for e in state.get("error_feedback") or []] or ["无法自动处理"])
    final = {"outcome": "HUMAN", "text": "这个问题我暂时无法给出可靠答复，已为您转接人工客服。",
             "reasons": reasons, "options": [], "draft": state.get("scheme")}
    return {"outcome": "HUMAN", "final": final,
            "trace": [ev("human_review", t0, None, reasons, "转人工", "human")]}


def finish(state):
    t0 = time.perf_counter()
    outcome = state.get("outcome") or "ANSWER"
    p = ParsedQuery.model_validate(state["parsed"]) if state.get("parsed") else None
    if outcome == "ANSWER":
        sc = state["scheme"]
        final = {"outcome": "ANSWER", "text": sc["text"], "options": sc["options"], "verify": state["verify"]}
    else:
        final = state["final"]
    mem = state["memory"]
    if p and p.order_id:
        memory.push_order(mem, p.order_id)
    c = _llm(state)
    memory.add_turn(mem, "user", state["question"], c)
    memory.add_turn(mem, "assistant", final["text"], c)
    if p and p.sub_queries:
        mem["last_intent"] = p.sub_queries[-1].intent
    memory.save_session(state["session_id"], state["user_id"], mem)
    if outcome == "ANSWER" and p and p.order_id:
        order = data.get_order(p.order_id)
        for i, r in enumerate(state["results"]):
            if r["intent"] in ("CHANGE", "REFUND"):
                memory.update_profile(state["user_id"], order, r["intent"],
                                      key=f"{state['session_id']}:{len(mem['turns'])}:{len(mem.get('archive', []))}:{i}")
    return {"outcome": outcome, "final": final, "memory": mem,
            "trace": [ev("finish", t0, None, {"outcome": outcome}, "写入记忆，本轮结束")]}


def _actionable(final):
    return [i for i, o in enumerate(final.get("options", []))
            if o.get("allowed") and o.get("type") in ("CHANGE", "REFUND")]


def await_confirm(state):
    t0 = time.perf_counter()
    final = state["final"]
    idx = _actionable(final)
    answer = interrupt({"kind": "confirm", "order_id": state["parsed"]["order_id"],
                        "options": [{"index": i, **final["options"][i]} for i in idx],
                        "message": "确认后将立即办理，操作不可撤销。"})
    if not (isinstance(answer, dict) and answer.get("confirm")):
        return {"outcome": "CANCELLED", "final": {**final, "outcome": "CANCELLED"},
                "trace": [ev("await_confirm", t0, None, answer, "乘客取消办理")]}
    i = int(answer.get("index", idx[0]))
    if i not in idx:
        return {"outcome": "CANCELLED", "final": {**final, "outcome": "CANCELLED"},
                "trace": [ev("await_confirm", t0, None, answer, "选择的方案不可办理", "error")]}
    confirm = {"index": i, "confirm_id": answer.get("confirm_id") or f"c-{uuid.uuid4().hex[:12]}"}
    return {"confirm": confirm, "trace": [ev("await_confirm", t0, None, confirm, "乘客已确认")]}


def tool_issue_ticket(state):
    t0 = time.perf_counter()
    cf = state["confirm"]
    r = state["results"][cf["index"]]
    args = {"order_id": state["parsed"]["order_id"], "action": r["intent"], "confirm_id": cf["confirm_id"]}
    if r["intent"] == "CHANGE":
        args.update(target_date=r["target_date"], target_flight=r["fee"]["target"]["flight_no"])
    res = tools.issue_ticket(args)
    if res["status"] == "CHANGED":
        text = f"已为您改签至 {res['target']['date']} {res['target']['flight_no']}，新票号 {res['ticket_no']}，共支付{res['amount']}元。"
    elif res["status"] == "REFUNDED":
        text = f"退票已受理，预计退款{res['amount']}元，将原路退回。"
    else:
        text = f"办理失败：{res['reason']}"
    outcome = "TICKETED" if res["status"] in ("CHANGED", "REFUNDED") else "HUMAN"
    return {"outcome": outcome, "final": {**state["final"], "outcome": outcome, "text": text, "ticket": res},
            "trace": [ev("tool_issue_ticket", t0, args, res, text, "ok" if outcome == "TICKETED" else "error")]}


# ---------------- 路由 ----------------
def after_understand(state):
    return "clarify" if state["parsed"].get("clarify") else "agent_node"


def after_clarify(state):
    return "human_review" if state.get("outcome") == "HUMAN" else "query_understand"


def after_agent(state):
    a = state["next_action"]["action"]
    if a == "finish":
        return "sub_done"
    if a in ("clarify", "human"):
        return {"clarify": "clarify", "human": "human_review"}[a]
    return "schema_validate"           # 未知工具名也交给校验节点报 UNKNOWN_TOOL


def after_validate(state):
    if state.get("error_feedback"):
        return "human_review" if state["tool_retries"] > S.get()["planner"]["max_tool_retries"] else "agent_node"
    return "checkpoint_gate"


def after_gate(state):
    if state.get("gate_reason"):
        return "human_review" if state.get("gate_blocks", 0) > 2 else "agent_node"
    return "tool_exec"


def after_tool(state):
    last = state["calls"][-1]
    if not last["ok"]:
        return "sub_done"
    p = state["parsed"]
    if (last["tool"] == "tool_rule_retrieve" and not state.get("stepped_back") and (p.get("fare_product") or p.get("airline"))
            and len(last["output"]["rules"]) < S.get()["query"]["step_back_min_hits"]):
        return "step_back"
    return "agent_node"


def after_sub(state):
    return "agent_node" if state["sub_idx"] < len(state["parsed"]["sub_queries"]) else "tool_scheme_gen"


def after_verify(state):
    if state["verify"]["passed"]:
        return "finish"
    return "retry_node" if state.get("retries", 0) < S.get()["verify"]["max_retries"] else "human_review"


def after_finish(state):
    return "await_confirm" if state["outcome"] == "ANSWER" and _actionable(state["final"]) else END


def after_confirm(state):
    return "tool_issue_ticket" if state.get("confirm") and state.get("outcome") != "CANCELLED" else END


def build_graph(checkpointer):
    b = StateGraph(AgentState)
    for fn in (user_input, memory_node, order_align, query_understand, clarify, agent_node, schema_validate,
               checkpoint_gate, tool_exec, step_back, sub_done, tool_scheme_gen, verify_node, retry_node,
               human_review, finish, await_confirm, tool_issue_ticket):
        b.add_node(fn.__name__, fn)
    b.add_edge(START, "user_input")
    b.add_edge("user_input", "memory_node")
    b.add_edge("memory_node", "order_align")
    b.add_edge("order_align", "query_understand")
    b.add_conditional_edges("query_understand", after_understand, ["clarify", "agent_node"])
    b.add_conditional_edges("clarify", after_clarify, ["human_review", "query_understand"])
    b.add_conditional_edges("agent_node", after_agent, ["sub_done", "clarify", "human_review", "schema_validate"])
    b.add_conditional_edges("schema_validate", after_validate, ["human_review", "agent_node", "checkpoint_gate"])
    b.add_conditional_edges("checkpoint_gate", after_gate, ["human_review", "agent_node", "tool_exec"])
    b.add_conditional_edges("tool_exec", after_tool, ["sub_done", "step_back", "agent_node"])
    b.add_edge("step_back", "agent_node")
    b.add_conditional_edges("sub_done", after_sub, ["agent_node", "tool_scheme_gen"])
    b.add_edge("tool_scheme_gen", "verify_node")
    b.add_conditional_edges("verify_node", after_verify, ["finish", "retry_node", "human_review"])
    b.add_edge("retry_node", "tool_scheme_gen")
    b.add_edge("human_review", "finish")
    b.add_conditional_edges("finish", after_finish, ["await_confirm", END])
    b.add_conditional_edges("await_confirm", after_confirm, ["tool_issue_ticket", END])
    b.add_edge("tool_issue_ticket", END)
    return b.compile(checkpointer=checkpointer)


# ---------------- 驱动 ----------------
def get_graph():
    global _graph
    if _graph is None:
        S.VAR_DIR.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(S.VAR_DIR / "checkpoints.db", check_same_thread=False)
        _graph = build_graph(SqliteSaver(conn))
    return _graph


def reset_graph():
    global _graph
    _graph = None
    _clients.clear()


def _cfg(session_id):
    return {"configurable": {"thread_id": session_id}, "recursion_limit": 80}


def pending_interrupt(session_id):
    snap = get_graph().get_state(_cfg(session_id))
    ints = list(getattr(snap, "interrupts", ()) or [i for t in snap.tasks for i in t.interrupts])
    return ints[0].value if ints else None


def _drive(inp, session_id, on_event=None):
    g = get_graph()
    events, pending = [], None
    for chunk in g.stream(inp, _cfg(session_id), stream_mode="updates"):
        for node, upd in chunk.items():
            if node == "__interrupt__":
                pending = upd[0].value
                if on_event:
                    on_event({"type": "interrupt", "payload": pending})
                continue
            for e in (upd or {}).get("trace", []):
                events.append(e)
                if on_event:
                    on_event({"type": "trace", "event": e})
    st = g.get_state(_cfg(session_id)).values
    mem = st.get("memory") or {}
    return {"session_id": session_id, "outcome": st.get("outcome"), "final": st.get("final"),
            "interrupt": pending, "trace": events, "parsed": st.get("parsed"), "context": st.get("context"),
            "results": st.get("results"), "verify": st.get("verify"),
            "memory": {k: mem.get(k) for k in ("order_stack", "summary", "turns", "last_intent")}}


def run_turn(session_id, user_id, message, order_id=None, mode=None, client=None, on_event=None):
    mode = mode or S.get()["mode"]
    _clients[session_id] = client if client is not None else llm_mod.get_client(mode)
    inp = {"session_id": session_id, "user_id": user_id, "question": message, "order_id": order_id, "mode": mode}
    with memory.session_lock(session_id):        # 整轮持锁：load_session → save_session 不被并发打断
        return _drive(inp, session_id, on_event)


def resume(session_id, payload, client=None, on_event=None):
    with memory.session_lock(session_id):
        if client is not None:
            _clients[session_id] = client
        elif session_id not in _clients:     # 服务重启后：按检查点里的模式重建客户端
            mode = get_graph().get_state(_cfg(session_id)).values.get("mode", "offline")
            _clients[session_id] = llm_mod.get_client(mode)
        return _drive(Command(resume=payload), session_id, on_event)
