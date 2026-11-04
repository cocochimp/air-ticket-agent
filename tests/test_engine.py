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


# ---- 评审修复补充 ----
def test_pct_half_up_golden():
    assert engine._pct(1250, 5) == 63      # 62.5 -> 63（银行家舍入会得到 62）
    assert engine._pct(1230, 5) == 62      # 61.5 -> 62
    assert engine._pct(1000, 0) == 0
    assert engine._pct(999, 100) == 999


def test_window_167h59m_is_h48():
    now = S.now()
    assert engine.window(now + timedelta(hours=167, minutes=59), now) == "h48"
    assert engine.window(now + timedelta(hours=168), now) == "d7"


def test_delay_under_2h_is_voluntary():
    o = _order(lambda o: data.get_flight(o["flight_no"], o["date"])["status"] == "delayed")
    data.set_flight_status(o["flight_no"], o["date"], "delayed", 119)
    r = engine.calc(o["order_id"], "REFUND")
    assert r["involuntary"] is False
    assert r["rule_id"] == f"{o['airline']}-{o['fare_product']}-RFD"


def test_target_cancelled_is_unavailable():
    o = _order(lambda o: o["user_id"] == "u_zhang")
    target = (date.fromisoformat(o["date"]) + timedelta(days=1)).isoformat()
    data.set_flight_status(o["flight_no"], target, "cancelled")
    with pytest.raises(engine.EngineError) as ei:
        engine.calc(o["order_id"], "CHANGE", target_date=target)
    assert ei.value.code == "TARGET_UNAVAILABLE"


def test_order_not_issued():
    o = _order(lambda o: o["user_id"] == "u_zhang")
    data.update_order(o["order_id"], status="REFUNDED")
    with pytest.raises(engine.EngineError) as ei:
        engine.calc(o["order_id"], "REFUND")
    assert ei.value.code == "ORDER_NOT_ISSUED"


def test_change_to_other_airline_rejected():
    o = _order(lambda o: o["user_id"] == "u_zhang")
    flight = data.get_flight(o["flight_no"], o["date"])
    target = (date.fromisoformat(o["date"]) + timedelta(days=1)).isoformat()
    other = next(f for f in data.list_flights(flight["origin"], flight["dest"], target)
                 if f["airline"] != o["airline"])
    with pytest.raises(engine.EngineError) as ei:
        engine.calc(o["order_id"], "CHANGE", target_date=target, target_flight=other["flight_no"])
    assert ei.value.code == "NO_TARGET_FLIGHT"


def test_change_to_same_flight_rejected():
    o = _order(lambda o: o["user_id"] == "u_zhang")
    with pytest.raises(engine.EngineError) as ei:
        engine.calc(o["order_id"], "CHANGE", target_date=o["date"])
    assert ei.value.code == "SAME_FLIGHT"
    assert "相同" in str(ei.value)


def test_missing_flight_row_is_data_missing():
    o = _order(lambda o: o["user_id"] == "u_zhang")
    with data.connect() as c:
        c.execute("DELETE FROM flights WHERE flight_no=? AND date=?", (o["flight_no"], o["date"]))
    with pytest.raises(engine.EngineError) as ei:
        engine.calc(o["order_id"], "REFUND")
    assert ei.value.code == "DATA_MISSING"


def test_missing_fee_rule_is_data_missing():
    o = _order(lambda o: o["user_id"] == "u_zhang")
    with data.connect() as c:
        c.execute("DELETE FROM fee_rules WHERE airline=? AND fare_product=?",
                  (o["airline"], o["fare_product"]))
    with pytest.raises(engine.EngineError) as ei:
        engine.calc(o["order_id"], "REFUND")
    assert ei.value.code == "DATA_MISSING"


def test_target_without_fare_product_is_data_missing():
    o = _order(lambda o: o["user_id"] == "u_zhang")
    target = (date.fromisoformat(o["date"]) + timedelta(days=1)).isoformat()
    with data.connect() as c:
        c.execute("UPDATE flights SET fares='{}' WHERE flight_no=? AND date=?", (o["flight_no"], target))
    with pytest.raises(engine.EngineError) as ei:
        engine.calc(o["order_id"], "CHANGE", target_date=target)
    assert ei.value.code == "DATA_MISSING"
