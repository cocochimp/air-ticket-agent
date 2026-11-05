import pytest
from air_agent import data, graph, llm

D = "20260926"
O1 = f"ORD{D}001"
FEE = lambda date: {"order_id": O1, "action": "CHANGE", "target_date": date}


@pytest.fixture(autouse=True)
def _fresh():
    data.init()
    graph.reset_graph()
    yield
    graph.reset_graph()


def act(tool, args=None, thought="..."):
    return {"thought": thought, "action": tool, "args": args or {}}


def test_schema_error_feedback_then_success():
    fake = llm.FakeLLM(react=[
        act("tool_rule_retrieve", {"query": "折扣票改签", "airline": "MU", "fare_product": "Y_DISCOUNT"}),
        act("tool_recalc_fee", FEE("后天")),                  # 相对日期 → TYPE 错误
        act("tool_recalc_fee", FEE("2026-09-28")),            # 带着反馈修正
        act("finish"),
    ])
    res = graph.run_turn("l1", "u_zhang", "改到后天", order_id=O1, mode="deepseek", client=fake)
    assert res["outcome"] == "ANSWER"
    sv = [e["status"] for e in res["trace"] if e["node"] == "schema_validate"]
    assert sv == ["ok", "error", "ok"]


def test_gate_blocks_fee_before_rules():
    fake = llm.FakeLLM(react=[
        act("tool_recalc_fee", FEE("2026-09-28")),            # 跳步：没查规则就算钱
        act("tool_rule_retrieve", {"query": "折扣票改签", "airline": "MU"}),
        act("tool_recalc_fee", FEE("2026-09-28")),
        act("finish"),
    ])
    res = graph.run_turn("l2", "u_zhang", "改到后天", order_id=O1, mode="deepseek", client=fake)
    gates = [e["status"] for e in res["trace"] if e["node"] == "checkpoint_gate"]
    assert gates[0] == "blocked" and res["outcome"] == "ANSWER"


def test_wrong_amount_text_triggers_retry_then_passes():
    steps = [act("tool_rule_retrieve", {"query": "折扣票改签", "airline": "MU", "fare_product": "Y_DISCOUNT"}),
             act("tool_recalc_fee", FEE("2026-09-28")), act("finish")]
    fake = llm.FakeLLM(react=steps, scheme_texts=["改签只需支付1元。", None])
    res = graph.run_turn("l3", "u_zhang", "改到后天", order_id=O1, mode="deepseek", client=fake)
    ns = [e["node"] for e in res["trace"]]
    assert "retry_node" in ns and res["outcome"] == "ANSWER"
    assert res["final"]["text"] != "改签只需支付1元。"


def test_llm_failure_falls_back_to_rules():
    fake = llm.FakeLLM(react=[llm.LLMError("timeout")] * 10)
    res = graph.run_turn("l4", "u_zhang", "改到后天", order_id=O1, mode="deepseek", client=fake)
    assert res["outcome"] == "ANSWER"
    assert all(e["output"]["source"] == "fallback" for e in res["trace"] if e["node"] == "agent_node")


def test_round_limit_goes_human():
    fake = llm.FakeLLM(react=[act("tool_flight_status", {"flight_no": "MU5101", "date": "2026-09-28"})] * 10)
    res = graph.run_turn("l5", "u_zhang", "改到后天", order_id=O1, mode="deepseek", client=fake)
    assert res["outcome"] == "HUMAN"
