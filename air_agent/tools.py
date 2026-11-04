"""Function Calling 工具：参数 Schema、校验与统一执行入口。"""
from typing import Literal
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator
from pydantic_core import PydanticCustomError
from . import data, engine

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
