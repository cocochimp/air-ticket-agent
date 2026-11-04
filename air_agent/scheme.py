"""方案生成：模板（规则模式）或 LLM（DeepSeek 模式）。金额一律回填引擎结果。"""
from .state import FeeItem, SchemeOption, SchemeOutput

STATUS_LABEL = {"scheduled": "计划正常", "delayed": "延误", "cancelled": "已取消"}


def _option(r: dict) -> tuple[SchemeOption, str]:
    refs = [x["id"] for x in r["rules"][:1]]
    fee = r["fee"]
    if r.get("error"):
        return (SchemeOption(type=r["intent"], title="暂时无法办理", allowed=False, rule_refs=[], notes=r["error"]),
                f"{r['text']}：{r['error']}。")
    if r["intent"] == "STATUS":
        st = r["status"]
        delay = f"，延误{st['delay_min']}分钟" if st["status"] == "delayed" else ""
        return (SchemeOption(type="STATUS", title="航班状态", notes=STATUS_LABEL[st["status"]]),
                f"航班{st['flight_no']}（{st['date']} {st['dep_time']}）当前{STATUS_LABEL[st['status']]}{delay}。")
    if r["intent"] == "POLICY" or fee is None:
        top = r["rules"][0] if r["rules"] else None
        if not top:
            return SchemeOption(type="POLICY", title="未找到规则", allowed=False), "没有检索到相关规则。"
        return (SchemeOption(type="POLICY", title=top["title"], rule_refs=[top["id"]]),
                f"根据《{top['title']}》：{top['content']}")
    refs = [fee["rule_id"]] if any(x["id"] == fee["rule_id"] for x in r["rules"]) else refs
    if not fee["allowed"]:
        return (SchemeOption(type=r["intent"], title="不可办理", allowed=False, rule_refs=refs, notes=fee["reason"]),
                f"很抱歉，{fee['reason']}（依据规则{refs[0] if refs else ''}）。")
    if r["intent"] == "CHANGE":
        t = fee["target"]
        items = [FeeItem(name="改期费", amount=fee["change_fee"]), FeeItem(name="票面差价", amount=fee["fare_diff"])]
        head = "因航班非自愿变动，" if fee["involuntary"] else f"当前处于{fee['window_label']}档，"
        text = (f"{head}改签至{t['date']} {t['flight_no']}（{t['dep_time']}）：改期费{fee['change_fee']}元，"
                f"票面差价{fee['fare_diff']}元，合计需支付{fee['total']}元。")
        return (SchemeOption(type="CHANGE", title=f"改签至 {t['date']} {t['flight_no']}", fee_items=items,
                             total=fee["total"], rule_refs=refs), text)
    items = [FeeItem(name="退票费", amount=fee["refund_fee"]), FeeItem(name="机场建设费退还", amount=fee["tax"])]
    head = "因航班非自愿变动，" if fee["involuntary"] else f"当前处于{fee['window_label']}档，"
    text = (f"{head}退票：退票费{fee['refund_fee']}元，退还机场建设费{fee['tax']}元，"
            f"预计退款{fee['refund_amount']}元。")
    return (SchemeOption(type="REFUND", title="退票", fee_items=items, total=fee["refund_amount"], rule_refs=refs),
            text)


def generate(results: list[dict], feedback: list[str] | None = None, llm=None, context: dict | None = None) -> SchemeOutput:
    opts, texts = [], []
    for r in results:
        o, t = _option(r)
        opts.append(o)
        texts.append(t)
    out = SchemeOutput(options=opts, text="".join(texts))
    if llm is not None:
        polished = llm.write_scheme(out, results, feedback or [], context or {})
        if polished:
            out = enforce_amounts(polished, out)
    return out


def enforce_amounts(candidate: SchemeOutput, reference: SchemeOutput) -> SchemeOutput:
    """LLM 只负责措辞：结构化字段一律用模板（引擎）值覆盖。"""
    return SchemeOutput(options=reference.options, text=candidate.text)
