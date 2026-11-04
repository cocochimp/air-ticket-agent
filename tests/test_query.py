from air_agent import data, query

D = "20260926"


def ctx(**kw):
    base = {"user_id": "u_zhang", "selected_order_id": None, "order_stack": [], "last_intent": None}
    return {**base, **kw}


def setup_function(_):
    data.init()


def test_resolve_dates():
    assert query.resolve_date("大后天") == "2026-09-29"
    assert query.resolve_date("下周六") == "2026-10-03"
    assert query.resolve_date("10月3号") == "2026-10-03"
    assert query.resolve_date("2026-10-05") == "2026-10-05"
    assert query.resolve_date("下个月初") is None


def test_multi_intent_split_and_dates():
    p = query.understand("帮我改到大后天，顺便看看退票多少钱", ctx(selected_order_id=f"ORD{D}001"))
    assert [s.intent for s in p.sub_queries] == ["CHANGE", "REFUND"]
    assert p.sub_queries[0].target_date == "2026-09-29"
    assert p.order_id == f"ORD{D}001" and p.airline == "MU" and p.fare_product == "Y_DISCOUNT"
    assert p.clarify is None and p.confidence >= 0.85


def test_return_leg_coreference():
    p = query.understand("回程那张能退吗", ctx(order_stack=[f"ORD{D}001"]))
    assert p.order_id == f"ORD{D}002"


def test_pronoun_uses_stack_top():
    p = query.understand("它退票要多少钱", ctx(order_stack=[f"ORD{D}003", f"ORD{D}001"]))
    assert p.order_id == f"ORD{D}003"


def test_ambiguous_order_asks_user():
    p = query.understand("我要退票", ctx())
    assert p.clarify["field"] == "order_id"
    assert len(p.clarify["options"]) >= 2
    assert p.confidence < 0.85


def test_change_without_date_asks_date():
    p = query.understand("我想改签", ctx(selected_order_id=f"ORD{D}001"))
    assert p.clarify["field"] == "target_date"
    assert p.clarify["options"][0]["value"] == "2026-09-27"


def test_unparseable_date_kept_raw():
    p = query.understand("改到下个月初", ctx(selected_order_id=f"ORD{D}001"))
    assert p.sub_queries[0].target_date == "下个月初" and p.clarify is None


def test_ellipsis_inherits_last_intent():
    p = query.understand("那大后天呢", ctx(selected_order_id=f"ORD{D}001", last_intent="CHANGE"))
    assert p.sub_queries[0].intent == "CHANGE" and p.sub_queries[0].target_date == "2026-09-29"


def test_policy_without_order_uses_synonyms():
    p = query.understand("东方云航特价票能退吗", ctx(user_id="u_wang_nobody"))
    assert p.sub_queries[0].intent == "POLICY"
    assert p.airline == "MU" and p.fare_product == "SPECIAL" and p.scenario == "VOLUNTARY_REFUND"


def test_scenario_from_flight_status():
    cancelled = next(o for o in data.list_orders("u_wang")
                     if data.get_flight(o["flight_no"], o["date"])["status"] == "cancelled")
    p = query.understand("航班取消了怎么办，我要退票", ctx(user_id="u_wang", selected_order_id=cancelled["order_id"]))
    assert p.scenario == "CANCEL"


def test_clarified_values_override():
    p = query.understand("改到下个月初", ctx(selected_order_id=f"ORD{D}001", forced_date="2026-10-02"))
    assert p.sub_queries[0].target_date == "2026-10-02"
    p = query.understand("嗯", ctx(selected_order_id=f"ORD{D}001", forced_intent="REFUND"))
    assert p.sub_queries[0].intent == "REFUND" and p.clarify is None


def test_step_back_and_hyde_text():
    p = query.understand("改到大后天", ctx(selected_order_id=f"ORD{D}001"))
    assert query.step_back_query(p, p.sub_queries[0]) == "经济舱自愿改签规则"
    assert "改期" in query.hyde_text(p, p.sub_queries[0])
