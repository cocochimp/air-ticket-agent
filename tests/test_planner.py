import httpx
import pytest
from air_agent import data, llm, planner, query
from air_agent import settings as S

D = "20260926"


def _state(question, **kw):
    data.init()
    ctx = {"user_id": "u_zhang", "selected_order_id": f"ORD{D}001", "order_stack": [], "last_intent": None,
           "forced_date": None, "forced_intent": None}
    p = query.understand(question, ctx)
    return {"parsed": p.model_dump(), "sub_idx": 0, "calls": [], "error_feedback": [], **kw}


def test_rule_plan_for_change():
    st = _state("改到大后天")
    seq = []
    for _ in range(4):
        a = planner.decide(st)
        seq.append(a["action"])
        st["calls"].append({"tool": a["action"], "args": a["args"], "ok": True, "output": {}})
    assert seq == ["tool_flight_status", "tool_rule_retrieve", "tool_recalc_fee", "finish"]


def test_rule_args_for_recalc():
    st = _state("改到大后天")
    st["calls"] = [{"tool": "tool_flight_status", "ok": True}, {"tool": "tool_rule_retrieve", "ok": True}]
    a = planner.decide(st)
    assert a["args"] == {"order_id": f"ORD{D}001", "action": "CHANGE", "target_date": "2026-09-29"}
    assert a["thought"]


def test_rule_retrieve_args_have_filters():
    st = _state("改到大后天")
    st["calls"] = [{"tool": "tool_flight_status", "ok": True}]
    a = planner.decide(st)
    assert a["args"]["airline"] == "MU" and a["args"]["fare_product"] == "Y_DISCOUNT"
    assert a["args"]["scenario"] == "VOLUNTARY_CHANGE"


def test_date_type_error_turns_into_clarify():
    st = _state("改到下个月初")
    st["error_feedback"] = [{"field": "target_date", "kind": "TYPE", "expected": "YYYY-MM-DD"}]
    a = planner.decide(st)
    assert a["action"] == "clarify" and a["args"]["field"] == "target_date"


def test_policy_plan():
    data.init()
    p = query.understand("东方云航特价票能退吗", {"user_id": "nobody", "selected_order_id": None,
                                               "order_stack": [], "last_intent": None,
           "forced_date": None, "forced_intent": None})
    st = {"parsed": p.model_dump(), "sub_idx": 0, "calls": [], "error_feedback": []}
    assert planner.decide(st)["action"] == "tool_rule_retrieve"


def test_fake_llm_drives_decision():
    st = _state("改到大后天")
    fake = llm.FakeLLM(react=[{"thought": "先算钱", "action": "tool_recalc_fee",
                               "args": {"order_id": f"ORD{D}001", "action": "CHANGE", "target_date": "后天"}}])
    a = planner.decide(st, fake)
    assert a["source"] == "llm" and a["args"]["target_date"] == "后天"


def test_llm_error_falls_back_to_rules():
    st = _state("改到大后天")
    a = planner.decide(st, llm.FakeLLM(react=[llm.LLMError("timeout")]))
    assert a["source"] == "fallback" and a["action"] == "tool_flight_status"


def test_deepseek_client_parses_tool_call(monkeypatch):
    S.update({"mode": "deepseek", "llm": {"api_key": "sk-test-000000"}})

    def handler(request):
        return httpx.Response(200, json={"choices": [{"message": {"content": "需要先查规则", "tool_calls": [
            {"id": "c1", "type": "function", "function": {"name": "tool_rule_retrieve",
                                                         "arguments": "{\"query\": \"改签\"}"}}]}}],
            "usage": {"prompt_tokens": 100, "completion_tokens": 20}})

    client = llm.DeepSeekClient(transport=httpx.MockTransport(handler))
    a = client.react([{"role": "user", "content": "x"}])
    assert a == {"thought": "需要先查规则", "action": "tool_rule_retrieve", "args": {"query": "改签"}}
    assert client.usage == {"calls": 1, "prompt_tokens": 100, "completion_tokens": 20}


def test_get_client_requires_key():
    S.update({"mode": "deepseek"})
    with pytest.raises(llm.LLMNotConfigured):
        llm.get_client("deepseek")
    assert llm.get_client("offline") is None


def test_malformed_payload_raises_llm_error():
    S.update({"mode": "deepseek", "llm": {"api_key": "sk-test-000000"}})
    client = llm.DeepSeekClient(transport=httpx.MockTransport(lambda r: httpx.Response(200, json={"oops": 1})))
    with pytest.raises(llm.LLMError):
        client.react([{"role": "user", "content": "x"}])
