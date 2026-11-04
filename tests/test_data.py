from air_agent import data, seed


def test_init_seeds_everything():
    data.init()
    orders = data.list_orders()
    assert len(orders) == 12
    assert {o["user_id"] for o in orders} == {"u_zhang", "u_li", "u_wang"}
    assert all(o["order_id"].startswith("ORD") and len(o["order_id"]) == 14 for o in orders)
    rules = data.list_rules()
    assert len(rules) == 4 * 6 * 2 + 4
    assert data.get_rule("MU-Y_DISCOUNT-CHG")["scenario"] == "VOLUNTARY_CHANGE"
    assert len(data.list_profiles()) == 3


def test_init_is_idempotent():
    data.init()
    data.init()
    assert len(data.list_orders()) == 12


def test_flight_status_update():
    data.init()
    o = data.list_orders("u_zhang")[0]
    data.set_flight_status(o["flight_no"], o["date"], "delayed", 150)
    f = data.get_flight(o["flight_no"], o["date"])
    assert f["status"] == "delayed" and f["delay_min"] == 150


def test_seed_scenarios_present():
    data.init()
    statuses = {data.get_flight(o["flight_no"], o["date"])["status"] for o in data.list_orders()}
    assert {"scheduled", "cancelled", "delayed"} <= statuses
    products = {o["fare_product"] for o in data.list_orders()}
    assert products == set(seed.PRODUCTS)


def test_rule_text_matches_fee_table():
    data.init()
    rule = data.get_rule("CA-Y-RFD")
    pct = data.get_fee_rule("CA", "Y")["refund"][0]
    assert f"{pct}%" in rule["content"]


# ---- 评审修复补充 ----
import pytest
from datetime import datetime
from air_agent import settings as S


def test_update_order_rejects_bad_fields():
    data.init()
    oid = data.list_orders()[0]["order_id"]
    with pytest.raises(ValueError):
        data.update_order(oid, user_id="u_x")
    with pytest.raises(ValueError):
        data.update_order(oid)
    data.update_order(oid, status="REFUNDED")
    assert data.get_order(oid)["status"] == "REFUNDED"


def test_set_flight_status_rejects_bad_status():
    data.init()
    o = data.list_orders()[0]
    with pytest.raises(ValueError):
        data.set_flight_status(o["flight_no"], o["date"], "exploded")


def test_empty_string_filters_are_not_ignored():
    data.init()
    assert data.list_orders("") == []
    assert data.list_flights(origin="") == []
    assert data.list_rules(airline="") == []
    assert len(data.list_orders(None)) == 12


def test_init_reseeds_on_new_day():
    data.init()
    assert data.list_orders()[0]["order_id"].startswith("ORD20260926")
    S.set_now(datetime(2026, 9, 27, 10, 0))
    data.init()
    orders = data.list_orders()
    assert len(orders) == 12
    assert all(o["order_id"].startswith("ORD20260927") for o in orders)


def test_init_same_day_keeps_changes():
    data.init()
    oid = data.list_orders()[0]["order_id"]
    data.update_order(oid, status="REFUNDED")
    data.init()
    assert data.get_order(oid)["status"] == "REFUNDED"


def test_force_init_is_deterministic():
    data.init(force=True)
    first = data.list_orders()
    data.init(force=True)
    assert data.list_orders() == first


def test_today_order_departure_crossing_midnight():
    S.set_now(datetime(2026, 9, 26, 22, 0))
    data.init()
    today_orders = [o for o in data.list_orders() if o["date"] == "2026-09-26"]
    assert today_orders
    assert data.get_flight(today_orders[0]["flight_no"], today_orders[0]["date"])["dep_time"] == "23:59"
