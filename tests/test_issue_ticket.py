import hashlib
import threading
from air_agent import data, engine, tools

D = "20260926"


def test_change_ticket_updates_order_and_is_idempotent():
    data.init()
    before = data.get_order(f"ORD{D}001")
    fee = engine.calc(before["order_id"], "CHANGE", target_date="2026-09-29")
    args = {"order_id": before["order_id"], "action": "CHANGE", "confirm_id": "confirm-001",
            "target_date": "2026-09-29"}
    first = tools.issue_ticket(args)
    second = tools.issue_ticket(args)
    after = data.get_order(before["order_id"])
    assert first["ticket_no"] == second["ticket_no"] and second["idempotent"] is True
    assert after["date"] == "2026-09-29" and after["ticket_no"] == first["ticket_no"]
    assert first["amount"] == fee["total"]


def test_refund_ticket():
    data.init()
    o = data.get_order(f"ORD{D}003")
    r = tools.issue_ticket({"order_id": o["order_id"], "action": "REFUND", "confirm_id": "confirm-002"})
    assert r["status"] == "REFUNDED" and data.get_order(o["order_id"])["status"] == "REFUNDED"


def test_not_allowed_is_rejected():
    data.init()
    gp = next(o for o in data.list_orders() if o["fare_product"] == "GROUP")
    r = tools.issue_ticket({"order_id": gp["order_id"], "action": "REFUND", "confirm_id": "confirm-003"})
    assert r["status"] == "REJECTED"


def test_ticket_no_is_deterministic():
    data.init()
    args = {"order_id": f"ORD{D}001", "action": "CHANGE", "confirm_id": "confirm-004", "target_date": "2026-09-29"}
    r = tools.issue_ticket(args)
    h = hashlib.sha1(f"confirm-004:{r['target']['flight_no']}".encode()).hexdigest()
    assert r["ticket_no"] == f"781-{int(h, 16) % 10**10:010d}"


def test_concurrent_same_confirm_issues_once(monkeypatch):
    data.init()
    calls = []
    orig = data.update_order
    monkeypatch.setattr(data, "update_order", lambda *a, **k: (calls.append(1), orig(*a, **k))[1])
    args = {"order_id": f"ORD{D}001", "action": "CHANGE", "confirm_id": "confirm-race", "target_date": "2026-09-29"}
    out, barrier = [], threading.Barrier(2)

    def run():
        barrier.wait()
        out.append(tools.issue_ticket(args))

    ths = [threading.Thread(target=run) for _ in range(2)]
    [t.start() for t in ths]
    [t.join() for t in ths]
    assert len(out) == 2 and sorted(r["idempotent"] for r in out) == [False, True]
    assert out[0]["ticket_no"] == out[1]["ticket_no"]
    assert len(calls) == 1
    with data.connect() as c:
        assert c.execute("SELECT COUNT(*) FROM tickets").fetchone()[0] == 1


def test_unexpected_exception_is_rejected(monkeypatch):
    data.init()
    monkeypatch.setattr(engine, "calc", lambda *a, **k: 1 / 0)
    r = tools.issue_ticket({"order_id": f"ORD{D}001", "action": "REFUND", "confirm_id": "confirm-boom"})
    assert r["status"] == "REJECTED" and r["reason"]
    with data.connect() as c:
        assert c.execute("SELECT COUNT(*) FROM tickets").fetchone()[0] == 0


def test_engine_error_is_rejected():
    data.init()
    r = tools.issue_ticket({"order_id": "ORD20260926999", "action": "REFUND", "confirm_id": "confirm-none"})
    assert r["status"] == "REJECTED"
