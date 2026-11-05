import pytest
from air_agent import data, engine, graph

D = "20260926"
O1, O3 = f"ORD{D}001", f"ORD{D}003"


@pytest.fixture(autouse=True)
def _fresh():
    data.init()
    graph.reset_graph()
    yield
    graph.reset_graph()


def nodes(res):
    return [e["node"] for e in res["trace"]]


def test_single_change_full_path():
    res = graph.run_turn("s1", "u_zhang", "帮我改到大后天", order_id=O1)
    assert res["outcome"] == "ANSWER"
    assert res["interrupt"]["kind"] == "confirm"
    expect = engine.calc(O1, "CHANGE", target_date="2026-09-29")["total"]
    assert res["final"]["options"][0]["total"] == expect
    ns = nodes(res)
    for n in ["user_input", "memory_node", "order_align", "query_understand", "agent_node", "schema_validate",
              "checkpoint_gate", "tool_exec", "sub_done", "tool_scheme_gen", "verify_node", "finish"]:
        assert n in ns
    tools_called = [e["output"]["action"] for e in res["trace"] if e["node"] == "agent_node"]
    assert tools_called == ["tool_flight_status", "tool_rule_retrieve", "tool_recalc_fee", "finish"]


def test_multi_intent_two_options():
    res = graph.run_turn("s2", "u_zhang", "改到大后天，顺便看看退票多少钱", order_id=O1)
    assert [o["type"] for o in res["final"]["options"]] == ["CHANGE", "REFUND"]
    assert nodes(res).count("sub_done") == 2


def test_clarify_order_then_resume():
    res = graph.run_turn("s3", "u_zhang", "我要退票")
    assert res["interrupt"]["kind"] == "clarify" and res["interrupt"]["field"] == "order_id"
    res = graph.resume("s3", {"value": O3})
    assert res["outcome"] == "ANSWER" and res["final"]["options"][0]["type"] == "REFUND"


def test_unparseable_date_schema_error_then_clarify():
    res = graph.run_turn("s4", "u_zhang", "改到下个月初", order_id=O1)
    errs = [e for e in res["trace"] if e["node"] == "schema_validate" and e["status"] == "error"]
    assert errs and "TYPE" in errs[0]["note"]
    assert res["interrupt"]["kind"] == "clarify" and res["interrupt"]["field"] == "target_date"
    res = graph.resume("s4", {"value": "2026-10-02"})
    assert res["outcome"] == "ANSWER"
    assert res["results"][0]["target_date"] == "2026-10-02"


def test_confirm_issues_ticket():
    graph.run_turn("s5", "u_zhang", "帮我改到大后天", order_id=O1)
    res = graph.resume("s5", {"confirm": True, "index": 0, "confirm_id": "confirm-s5"})
    assert res["outcome"] == "TICKETED"
    assert data.get_order(O1)["date"] == "2026-09-29"
    assert res["interrupt"] is None


def test_cancel_confirm():
    graph.run_turn("s6", "u_zhang", "帮我改到大后天", order_id=O1)
    res = graph.resume("s6", {"confirm": False})
    assert res["outcome"] == "CANCELLED" and data.get_order(O1)["date"] != "2026-09-29"


def test_follow_up_uses_memory():
    graph.run_turn("s7", "u_zhang", "帮我改到大后天", order_id=O1)
    res = graph.run_turn("s7", "u_zhang", "那退票呢")
    assert res["parsed"]["order_id"] == O1
    assert res["final"]["options"][0]["type"] == "REFUND"
    assert res["memory"]["order_stack"][0] == O1


def test_off_topic_goes_human():
    res = graph.run_turn("s8", "u_wang", "宠物能带上飞机吗")
    # query 模块对猜出来的意图先追问；按「了解规则」作答后检索不到依据 → 自我修正 → 转人工
    assert res["interrupt"]["kind"] == "clarify" and res["interrupt"]["field"] == "intent"
    res = graph.resume("s8", {"value": "POLICY"})
    assert res["outcome"] == "HUMAN"
    assert any("依据规则" in r for r in res["final"]["reasons"])
    assert "retry_node" in nodes(res)


def test_cancelled_flight_is_free_in_answer():
    o = next(x for x in data.list_orders("u_wang")
             if data.get_flight(x["flight_no"], x["date"])["status"] == "cancelled")
    res = graph.run_turn("s9", "u_wang", "航班取消了，帮我退票", order_id=o["order_id"])
    opt = res["final"]["options"][0]
    assert opt["total"] == o["fare"] + o["tax"] and "GEN-CANCEL" in opt["rule_refs"]


def test_resume_after_restart():
    graph.run_turn("s10", "u_zhang", "我要退票")
    graph.reset_graph()                 # 模拟服务重启：新连接，从 SQLite 检查点恢复
    assert graph.pending_interrupt("s10")["kind"] == "clarify"
    res = graph.resume("s10", {"value": O3})
    assert res["outcome"] == "ANSWER"


def test_off_topic_with_selected_order_goes_human():
    res = graph.run_turn("s11", "u_zhang", "宠物能带上飞机吗", order_id=O1)
    assert res["interrupt"]["kind"] == "clarify"
    res = graph.resume("s11", {"value": "POLICY"})
    assert res["outcome"] == "HUMAN"
