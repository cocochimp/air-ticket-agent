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
