from datetime import date, timedelta
from air_agent import data, tools


def test_missing_target_date_for_change():
    data.init()
    oid = data.list_orders()[0]["order_id"]
    args, errs = tools.validate("tool_recalc_fee", {"order_id": oid, "action": "CHANGE"})
    assert args is None
    assert errs[0]["field"] == "target_date" and errs[0]["kind"] == "MISSING"


def test_relative_date_rejected_as_type_error():
    args, errs = tools.validate("tool_recalc_fee",
                                {"order_id": "ORD20260926001", "action": "CHANGE", "target_date": "后天"})
    assert errs[0]["field"] == "target_date" and errs[0]["kind"] == "TYPE"
    assert "YYYY-MM-DD" in errs[0]["expected"]


def test_enum_error():
    _, errs = tools.validate("tool_rule_retrieve", {"query": "退票", "cabin": "经济舱"})
    assert errs[0]["kind"] == "ENUM" and errs[0]["field"] == "cabin"


def test_bad_order_id_format():
    _, errs = tools.validate("tool_recalc_fee",
                             {"order_id": "https://x/ORD1", "action": "REFUND"})
    assert errs[0]["kind"] == "TYPE"


def test_unknown_tool():
    _, errs = tools.validate("tool_magic", {})
    assert errs[0]["kind"] == "UNKNOWN_TOOL"


def test_run_recalc_fee_ok_and_engine_error():
    data.init()
    o = data.list_orders("u_zhang")[0]
    ok = tools.run_tool("tool_recalc_fee", {"order_id": o["order_id"], "action": "REFUND"})
    assert ok["ok"] and ok["output"]["refund_amount"] > 0
    bad = tools.run_tool("tool_recalc_fee", {"order_id": "ORD99999999999", "action": "REFUND"})
    assert not bad["ok"] and bad["kind"] == "ENGINE"


def test_run_flight_status():
    data.init()
    o = data.list_orders("u_wang")[0]
    r = tools.run_tool("tool_flight_status", {"flight_no": o["flight_no"], "date": o["date"]})
    assert r["ok"] and r["output"]["status"] in {"scheduled", "delayed", "cancelled"}


def test_llm_specs_cover_react_tools():
    names = [t["function"]["name"] for t in tools.llm_specs()]
    assert names == ["tool_flight_status", "tool_rule_retrieve", "tool_recalc_fee", "finish"]
