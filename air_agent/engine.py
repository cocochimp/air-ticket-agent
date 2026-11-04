"""运价引擎：改签费、差价、退票费。系统里所有金额都只能从这里来。"""
from datetime import datetime
from . import data, seed
from . import settings as S


class EngineError(Exception):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


def departure(flight) -> datetime:
    return datetime.fromisoformat(f"{flight['date']}T{flight['dep_time']}")


def window(dep: datetime, at: datetime) -> str:
    hours = (dep - at).total_seconds() / 3600
    if hours < 0:
        return "after"
    if hours < 4:
        return "lt4"
    if hours < 48:
        return "h4"
    if hours < 168:
        return "h48"
    return "d7"


def involuntary_reason(flight):
    if flight["status"] == "cancelled":
        return "GEN-CANCEL"
    if flight["status"] == "delayed" and flight["delay_min"] >= 120:
        return "GEN-DELAY"
    return None


def calc(order_id: str, action: str, target_date: str | None = None, target_flight: str | None = None) -> dict:
    order = data.get_order(order_id)
    if not order:
        raise EngineError("ORDER_NOT_FOUND", f"订单不存在：{order_id}")
    if order["status"] != "ISSUED":
        raise EngineError("ORDER_NOT_ISSUED", f"订单状态为 {order['status']}，不能办理退改")
    flight = data.get_flight(order["flight_no"], order["date"])
    now = S.now()
    w = window(departure(flight), now)
    inv = involuntary_reason(flight)
    rates = data.get_fee_rule(order["airline"], order["fare_product"])
    fare, tax = order["fare"], order["tax"]
    base = {"order_id": order_id, "action": action, "currency": "CNY", "fare": fare, "tax": tax,
            "window": w, "window_label": seed.WINDOW_LABEL[w], "involuntary": bool(inv),
            "flight_status": flight["status"], "delay_min": flight["delay_min"], "allowed": True,
            "reason": "", "rate_pct": 0}

    if action == "CHANGE":
        if not target_date:
            raise EngineError("MISSING_TARGET_DATE", "改签必须提供 target_date（YYYY-MM-DD）")
        tf = data.get_flight(target_flight or order["flight_no"], target_date)
        if not tf or (tf["origin"], tf["dest"]) != (flight["origin"], flight["dest"]):
            raise EngineError("NO_TARGET_FLIGHT", f"{target_date} 没有可改签的航班")
        if tf["status"] == "cancelled" or departure(tf) <= now:
            raise EngineError("TARGET_UNAVAILABLE", f"{tf['flight_no']} {target_date} 不可改签")
        target = {"flight_no": tf["flight_no"], "date": target_date, "dep_time": tf["dep_time"],
                  "fare": tf["fares"][order["fare_product"]]}
        base.update(target=target, change_fee=0, fare_diff=0, total=0)
        if inv:
            return {**base, "rule_id": inv, "reason": "非自愿改签，免改期费、免差价"}
        rule_id = f"{order['airline']}-{order['fare_product']}-CHG"
        if rates["change"] is None:
            return {**base, "rule_id": rule_id, "allowed": False, "reason": "该票价产品不得自愿改签"}
        pct = rates["change"][seed.WINDOWS.index(w)]
        fee = round(fare * pct / 100)
        diff = max(0, target["fare"] - fare)
        return {**base, "rule_id": rule_id, "rate_pct": pct, "change_fee": fee, "fare_diff": diff,
                "total": fee + diff}

    if action == "REFUND":
        base.update(refund_fee=0, refund_amount=0)
        if inv:
            return {**base, "rule_id": inv, "refund_amount": fare + tax, "reason": "非自愿退票，全额退款"}
        rule_id = f"{order['airline']}-{order['fare_product']}-RFD"
        if rates["refund"] is None:
            return {**base, "rule_id": rule_id, "allowed": False, "reason": "该票价产品不得自愿退票"}
        pct = rates["refund"][seed.WINDOWS.index(w)]
        fee = round(fare * pct / 100)
        return {**base, "rule_id": rule_id, "rate_pct": pct, "refund_fee": fee,
                "refund_amount": fare - fee + tax}

    raise EngineError("BAD_ACTION", f"不支持的操作：{action}")
