from datetime import date, timedelta
import pytest
from air_agent import data, engine, seed
from air_agent import settings as S


def _order(pred):
    data.init()
    return next(o for o in data.list_orders() if pred(o))


def test_window_boundaries():
    now = S.now()
    assert engine.window(now + timedelta(days=7), now) == "d7"
    assert engine.window(now + timedelta(hours=48), now) == "h48"
    assert engine.window(now + timedelta(hours=4), now) == "h4"
    assert engine.window(now + timedelta(hours=3, minutes=59), now) == "lt4"
    assert engine.window(now - timedelta(minutes=1), now) == "after"


def test_change_fee_and_fare_diff():
    o = _order(lambda o: o["user_id"] == "u_zhang" and o["fare_product"] == "Y_DISCOUNT"
               and o["flight_no"].startswith("MU51") and o["flight_no"].endswith("01"))
    target = (date.fromisoformat(o["date"]) + timedelta(days=1)).isoformat()
    r = engine.calc(o["order_id"], "CHANGE", target_date=target)
    w = r["window"]
    pct = data.get_fee_rule("MU", "Y_DISCOUNT")["change"][seed.WINDOWS.index(w)]
    new_fare = data.get_flight(o["flight_no"], target)["fares"]["Y_DISCOUNT"]
    assert r["allowed"] and not r["involuntary"]
    assert r["change_fee"] == round(o["fare"] * pct / 100)
    assert r["fare_diff"] == max(0, new_fare - o["fare"])
    assert r["total"] == r["change_fee"] + r["fare_diff"]
    assert r["rule_id"] == "MU-Y_DISCOUNT-CHG"


def test_refund_amount_includes_tax():
    o = _order(lambda o: o["fare_product"] == "C")
    r = engine.calc(o["order_id"], "REFUND")
    assert r["refund_amount"] == o["fare"] - r["refund_fee"] + o["tax"]
    assert r["rule_id"] == "CA-C-RFD"


def test_cancelled_flight_is_free():
    o = _order(lambda o: data.get_flight(o["flight_no"], o["date"])["status"] == "cancelled")
    target = (date.fromisoformat(o["date"]) + timedelta(days=1)).isoformat()
    ch = engine.calc(o["order_id"], "CHANGE", target_date=target)
    rf = engine.calc(o["order_id"], "REFUND")
    assert ch["involuntary"] and ch["total"] == 0 and ch["rule_id"] == "GEN-CANCEL"
    assert rf["refund_fee"] == 0 and rf["refund_amount"] == o["fare"] + o["tax"]


def test_delay_over_2h_is_free():
    o = _order(lambda o: data.get_flight(o["flight_no"], o["date"])["status"] == "delayed")
    r = engine.calc(o["order_id"], "REFUND")
    assert r["involuntary"] and r["refund_fee"] == 0 and r["rule_id"] == "GEN-DELAY"


def test_special_cannot_change_group_cannot_refund():
    sp = _order(lambda o: o["fare_product"] == "SPECIAL")
    gp = _order(lambda o: o["fare_product"] == "GROUP")
    target = (date.fromisoformat(sp["date"]) + timedelta(days=1)).isoformat()
    assert engine.calc(sp["order_id"], "CHANGE", target_date=target)["allowed"] is False
    assert engine.calc(gp["order_id"], "REFUND")["allowed"] is False


def test_no_show_uses_after_window():
    o = _order(lambda o: o["date"] < S.now().date().isoformat())
    r = engine.calc(o["order_id"], "REFUND")
    assert r["window"] == "after"


def test_errors():
    data.init()
    with pytest.raises(engine.EngineError, match="订单不存在"):
        engine.calc("ORD00000000000", "REFUND")
    o = data.list_orders()[0]
    with pytest.raises(engine.EngineError, match="target_date"):
        engine.calc(o["order_id"], "CHANGE")
    with pytest.raises(engine.EngineError, match="没有可改签的航班"):
        engine.calc(o["order_id"], "CHANGE", target_date="2030-01-01")
