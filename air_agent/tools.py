"""Function Calling 工具：参数 Schema、校验与统一执行入口。"""
import hashlib
import json
from typing import Literal
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator
from pydantic_core import PydanticCustomError
from . import data, engine
from . import settings as S

DATE = r"^\d{4}-\d{2}-\d{2}$"
FLIGHT = r"^[A-Z0-9]{2}\d{3,4}$"
ORDER = r"^ORD\d{11}$"
Cabin = Literal["F", "C", "Y"]
FareProduct = Literal["F", "C", "Y", "Y_DISCOUNT", "SPECIAL", "GROUP"]
Scenario = Literal["VOLUNTARY_CHANGE", "VOLUNTARY_REFUND", "CANCEL", "DELAY", "NO_SHOW", "FARE_DIFF"]


class _Args(BaseModel):
    model_config = ConfigDict(extra="forbid")


class FlightStatusArgs(_Args):
    """查询航班实时状态（正常/延误/取消），用于判断是否属于非自愿退改。"""
    flight_no: str = Field(pattern=FLIGHT, description="航班号，如 MU5101")
    date: str = Field(pattern=DATE, description="航班日期，YYYY-MM-DD 绝对日期")


class RuleRetrieveArgs(_Args):
    """检索退改签规则。"""
    query: str = Field(min_length=1, max_length=200, description="检索问题")
    airline: str | None = Field(default=None, pattern=r"^[A-Z0-9]{2}$", description="航司二字码")
    cabin: Cabin | None = None
    fare_product: FareProduct | None = None
    scenario: Scenario | None = None


class RecalcFeeArgs(_Args):
    """调用运价引擎计算改签费/差价或退票费。金额只能来自这个工具。"""
    order_id: str = Field(pattern=ORDER, description="订单号，ORD + 11 位数字")
    action: Literal["CHANGE", "REFUND"]
    target_date: str | None = Field(default=None, pattern=DATE, description="改签目标日期，YYYY-MM-DD 绝对日期")
    target_flight: str | None = Field(default=None, pattern=FLIGHT)

    @model_validator(mode="after")
    def _need_date(self):
        if self.action == "CHANGE" and not self.target_date:
            raise PydanticCustomError("missing_target_date", "改签必须提供 target_date")
        return self


class FinishArgs(_Args):
    """当前子诉求所需信息已齐全，进入方案生成。"""
    reason: str = Field(default="", max_length=200)


class IssueTicketArgs(_Args):
    order_id: str = Field(pattern=ORDER)
    action: Literal["CHANGE", "REFUND"]
    confirm_id: str = Field(pattern=r"^[\w-]{6,64}$")
    target_date: str | None = Field(default=None, pattern=DATE)
    target_flight: str | None = Field(default=None, pattern=FLIGHT)


SCHEMAS = {"tool_flight_status": FlightStatusArgs, "tool_rule_retrieve": RuleRetrieveArgs,
           "tool_recalc_fee": RecalcFeeArgs, "finish": FinishArgs, "tool_issue_ticket": IssueTicketArgs}
REACT_TOOLS = ["tool_flight_status", "tool_rule_retrieve", "tool_recalc_fee", "finish"]
EXPECTED = {DATE: "YYYY-MM-DD 绝对日期（先把「后天」等相对日期换算好）", FLIGHT: "航班号，如 MU5101",
            ORDER: "订单号，ORD + 11 位数字，不要传链接"}


def _kind(err) -> str:
    t = err["type"]
    if t.startswith("missing"):
        return "MISSING"
    if t in {"literal_error", "enum"}:
        return "ENUM"
    return "TYPE"


def _expected(err) -> str:
    ctx = err.get("ctx") or {}
    if "pattern" in ctx:
        return EXPECTED.get(ctx["pattern"], f"匹配 {ctx['pattern']}")
    if "expected" in ctx:
        return f"取值之一：{ctx['expected']}"
    if err["type"] == "missing_target_date":
        return EXPECTED[DATE]
    return err["msg"]


def _unknown_tool_error(tool) -> dict:
    return {"field": "tool", "kind": "UNKNOWN_TOOL", "expected": "、".join(REACT_TOOLS),
            "got": tool, "message": f"没有工具 {tool}"}


def validate(tool: str, args: dict):
    """返回 (校验后的模型, 错误列表)。错误项：{field, kind, expected, got, message}。"""
    schema = SCHEMAS.get(tool)
    if schema is None:
        return None, [_unknown_tool_error(tool)]
    try:
        return schema.model_validate(args or {}), []
    except ValidationError as e:
        errs = []
        for err in e.errors():
            field = ".".join(str(x) for x in err["loc"]) or ("target_date" if "target_date" in err["msg"] else "args")
            kind = _kind(err)
            errs.append({"field": field, "kind": kind, "expected": _expected(err),
                         "got": None if kind == "MISSING" else err.get("input"),
                         "message": f"{field}：{_expected(err)}"})
        return None, errs


def run_tool(tool: str, args: dict, boost: dict | None = None) -> dict:
    """统一执行入口：{ok, output} 或 {ok: False, kind, errors|error}。"""
    if tool not in REACT_TOOLS:       # tool_issue_ticket 只能走确认流程，不允许经此入口执行
        return {"ok": False, "kind": "UNKNOWN_TOOL", "errors": [_unknown_tool_error(tool)]}
    model, errs = validate(tool, args)
    if errs:
        return {"ok": False, "kind": errs[0]["kind"], "errors": errs}
    a = model.model_dump()
    try:
        if tool == "tool_flight_status":
            f = data.get_flight(a["flight_no"], a["date"])
            if not f:
                return {"ok": False, "kind": "ENGINE", "error": {"code": "FLIGHT_NOT_FOUND",
                                                                  "message": f"查不到航班 {a['flight_no']} {a['date']}"}}
            return {"ok": True, "output": {k: f[k] for k in ("flight_no", "date", "dep_time", "status", "delay_min")}}
        if tool == "tool_rule_retrieve":
            from . import retrieval
            filters = {k: a[k] for k in ("airline", "cabin", "fare_product", "scenario") if a[k]}
            return {"ok": True, "output": retrieval.search(a["query"], filters, boost or {})}
        if tool == "tool_recalc_fee":
            return {"ok": True, "output": engine.calc(a["order_id"], a["action"], a["target_date"], a["target_flight"])}
        if tool == "finish":
            return {"ok": True, "output": {"reason": a["reason"]}}
    except engine.EngineError as e:
        return {"ok": False, "kind": "ENGINE", "error": {"code": e.code, "message": str(e)}}
    except Exception as e:
        return {"ok": False, "kind": "ENGINE", "error": {"code": "INTERNAL", "message": f"{type(e).__name__}: {e}"}}
    return {"ok": False, "kind": "UNKNOWN_TOOL", "errors": [_unknown_tool_error(tool)]}


def llm_specs() -> list[dict]:
    """给 LLM 的 OpenAI 格式工具定义。"""
    out = []
    for name in REACT_TOOLS:
        schema = SCHEMAS[name]
        out.append({"type": "function", "function": {"name": name, "description": schema.__doc__.strip(),
                                                      "parameters": schema.model_json_schema()}})
    return out


def _ticket_no(confirm_id: str, flight_no: str) -> str:
    """由 confirm_id 和目标航班确定性生成票号（不用 hash()：它每个进程的盐不同）。"""
    h = hashlib.sha1(f"{confirm_id}:{flight_no}".encode()).hexdigest()
    return f"781-{int(h, 16) % 10**10:010d}"


def issue_ticket(args: dict) -> dict:
    """出票/退票。金额在出票时由引擎重算；同一 confirm_id 只办理一次。永不抛异常，失败返回 REJECTED。"""
    model, errs = validate("tool_issue_ticket", args)
    if errs:
        return {"status": "REJECTED", "reason": errs[0]["message"]}
    a = model.model_dump()
    try:
        # 幂等检查、改订单、写 tickets 在同一个写事务里：并发的同一 confirm_id 只会办理一次
        with data.transaction() as c:
            row = c.execute("SELECT result FROM tickets WHERE confirm_id=?", (a["confirm_id"],)).fetchone()
            if row:
                return {**json.loads(row["result"]), "idempotent": True}
            fee = engine.calc(a["order_id"], a["action"], a["target_date"], a["target_flight"])
            if not fee["allowed"]:
                return {"status": "REJECTED", "reason": fee["reason"]}
            order = data.get_order(a["order_id"])
            if a["action"] == "CHANGE":
                t = fee["target"]
                ticket_no = _ticket_no(a["confirm_id"], t["flight_no"])
                data.update_order(a["order_id"], conn=c, flight_no=t["flight_no"], date=t["date"],
                                  fare=max(order["fare"], t["fare"]), ticket_no=ticket_no)
                result = {"status": "CHANGED", "action": "CHANGE", "order_id": a["order_id"],
                          "ticket_no": ticket_no, "amount": fee["total"], "target": t}
            else:
                data.update_order(a["order_id"], conn=c, status="REFUNDED")
                result = {"status": "REFUNDED", "action": "REFUND", "order_id": a["order_id"],
                          "ticket_no": order["ticket_no"], "amount": fee["refund_amount"]}
            c.execute("INSERT INTO tickets VALUES(?,?,?,?,?)",
                      (a["confirm_id"], a["order_id"], a["action"],
                       json.dumps(result, ensure_ascii=False), S.now().isoformat()))
        return {**result, "idempotent": False}
    except engine.EngineError as e:
        return {"status": "REJECTED", "reason": str(e)}
    except Exception as e:
        return {"status": "REJECTED", "reason": f"出票失败：{type(e).__name__}: {e}"}
