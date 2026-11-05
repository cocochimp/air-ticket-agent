"""LangGraph 状态与各模块之间传递的数据模型。"""
from typing import Annotated, Literal, TypedDict
from pydantic import BaseModel, Field

Intent = Literal["CHANGE", "REFUND", "POLICY", "STATUS"]
def trace_reducer(old: list | None, new: list | None) -> list:
    """每轮从 user_input 开始重置 trace；resume 时继续追加。"""
    new = new or []
    if new and new[0].get("node") == "user_input":
        return list(new)
    return (old or []) + new


INTENT_LABEL = {"CHANGE": "改签", "REFUND": "退票", "POLICY": "政策咨询", "STATUS": "航班状态"}


class SubQuery(BaseModel):
    intent: Intent
    text: str
    target_date: str | None = None


class ParsedQuery(BaseModel):
    intents: list[Intent] = Field(default_factory=list)
    sub_queries: list[SubQuery] = Field(default_factory=list)
    order_id: str | None = None
    airline: str | None = None
    cabin: str | None = None
    fare_product: str | None = None
    scenario: str | None = None
    missing: list[str] = Field(default_factory=list)
    rewritten: str = ""
    step_back: str | None = None
    confidence: float = 1.0
    clarify: dict | None = None      # {"field", "question", "options": [{"label", "value"}]}


class FeeItem(BaseModel):
    name: str
    amount: int


class SchemeOption(BaseModel):
    type: Intent
    title: str
    allowed: bool = True
    fee_items: list[FeeItem] = Field(default_factory=list)
    total: int | None = None
    rule_refs: list[str] = Field(default_factory=list)
    notes: str = ""


class SchemeOutput(BaseModel):
    options: list[SchemeOption]
    text: str
    texts: list[str] = Field(default_factory=list)   # 与 options 一一对应的子方案文案，供 verify 逐项校验


class AgentState(TypedDict, total=False):
    # 输入
    session_id: str
    user_id: str
    order_id: str | None
    question: str
    mode: str                     # offline | deepseek
    # 记忆与理解
    context: dict                 # memory_node 注入的上下文与 Token 预算
    parsed: dict                  # ParsedQuery
    # ReAct
    sub_idx: int
    calls: list[dict]             # 当前子诉求的工具调用记录
    results: list[dict]           # 每个子诉求的结果：{intent, status, rules, funnel, fee}
    next_action: dict             # {thought, action, args, source}
    error_feedback: list[dict]
    rounds: int
    tool_retries: int
    gate_reason: str
    # 生成与校验
    scheme: dict
    verify: dict
    retries: int
    feedback: list[str]
    # 结果
    outcome: str                  # ANSWER | CLARIFY | HUMAN | TICKETED | CANCELLED
    final: dict
    # 图运行期辅助字段
    memory: dict
    clarified: dict               # 追问得到的值：{order_id | target_date | intent: value}
    clarify_req: dict | None      # agent_node 发起的追问
    clarify_count: int
    gate_blocks: int
    stepped_back: bool
    confirm: dict | None
    trace: Annotated[list[dict], trace_reducer]
