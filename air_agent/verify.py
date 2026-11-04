"""工作五：金额逐项后验、论断事实核查、引用合法性，计算置信度。

校验口径（防 LLM 幻觉）：
- 文案先 NFKC 归一化，统一千分位、￥/RMB/块 等写法为「N元」。
- 有标签的金额（改期费/差价/合计/退票费/机场建设费/退款…）必须等于引擎对应字段。
- 无标签的金额必须是引擎展示字段之一；有小数、中文数字一律判错。
- 规则原文自带的金额只对 POLICY 方案开放，且金额前后 10 个字必须逐字出现在被引用的规则里。
- 论断关键词按短语（关键词 + 后 2 个汉字）在单条被引用规则中核对；百分比必须等于引擎 rate_pct。
- 只要有一句无依据论断，置信度为 0。
"""
import re
import unicodedata

AMOUNT = re.compile(r"(\d+(?:\.\d+)?)\s*元")
CN_AMOUNT = re.compile(r"[零一二三四五六七八九十百千万两]+\s*(?:元|块)")
CLAIM_KEYS = re.compile(r"免费|免收|全额|不得|不可|可以|\d+次|\d+%|会员|积分|赔偿|补偿|无需|不用|不收|包退|包改"
                        r"|随便|不限|保证|承诺|退差|百分之|打折|折扣")
LABEL = re.compile(r"(改期费|票面差价|差价|票面价|合计|共需支付|需支付|退票费|手续费|机场建设费|退款|退还|退回)"
                   r"([^0-9。；，;,!?！？]{0,8}?)(\d+(?:\.\d+)?)\s*元")
LABEL_FIELD = {  # 标签 -> (适用的子诉求类型, 引擎字段)
    "改期费": ("CHANGE", "change_fee"), "票面差价": ("CHANGE", "fare_diff"), "差价": ("CHANGE", "fare_diff"),
    "合计": ("CHANGE", "total"), "共需支付": ("CHANGE", "total"), "需支付": ("CHANGE", "total"),
    "退票费": ("REFUND", "refund_fee"), "手续费": ("REFUND", "refund_fee"),
    "机场建设费": (None, "tax"), "票面价": (None, "fare"),
    "退款": ("REFUND", "refund_amount"), "退还": ("REFUND", "refund_amount"), "退回": ("REFUND", "refund_amount"),
}
REFUND_WORDS = {"退款", "退还", "退回"}
SHOWN_FIELDS = ("change_fee", "fare_diff", "total", "refund_fee", "refund_amount", "tax")
SPLIT = re.compile(r"[。！？；!?;]")
CJK_TAIL = re.compile(r"[一-鿿]{0,2}")


def _norm(text: str) -> str:
    t = unicodedata.normalize("NFKC", text)
    t = re.sub(r"(?<=\d),(?=\d)", "", t)
    t = re.sub(r"(?:¥|RMB|CNY|人民币)\s*(\d+(?:\.\d+)?)(?:\s*元)?", r"\1元", t, flags=re.I)
    return re.sub(r"(\d+(?:\.\d+)?)\s*块(?:钱)?", r"\1元", t)


def _num(fee, key):
    v = (fee or {}).get(key)
    return v if isinstance(v, (int, float)) and not isinstance(v, bool) else None


def _rule_text(rule) -> str:
    return _norm(f"{rule['title']}{rule['content']}")


def _rule_corpus(results):
    return "".join(f"{x['title']}{x['content']}" for r in results for x in r.get("rules", []))


def _cited_rules(unit, only_policy=False):
    ids = {ref for o in unit["options"] if not only_policy or o["type"] == "POLICY"
           for ref in o.get("rule_refs", [])}
    return [_rule_text(x) for r in unit["results"] for x in r.get("rules", []) if x["id"] in ids]


def _fee_results(unit):
    return [r for r in unit["results"] if r.get("fee") and r["intent"] in ("CHANGE", "REFUND")]


def _in_rule(text, start, end, rule_texts) -> bool:
    left, right = text.rfind("。", 0, start), text.find("。", end)      # 窗口不越过所在句的句号
    lo = max(0, start - 10, left + 1)
    hi = min(len(text), end + 10, right + 1 if right >= 0 else len(text))
    win = text[lo:hi]
    if "》:" in win:
        win = win.split("》:")[-1]
    return any(win in rt for rt in rule_texts)


def _check_amounts(unit):
    text, errors, bad = unit["text"], [], []
    fee_res = _fee_results(unit)
    shown = {int(v) for r in fee_res for k in SHOWN_FIELDS if (v := _num(r["fee"], k)) is not None}
    policy_texts = _cited_rules(unit, only_policy=True)
    labelled = {m.start(3): m for m in LABEL.finditer(text)}
    if CN_AMOUNT.search(text):
        errors.append("金额必须使用阿拉伯数字并来自工具结果")
    for m in AMOUNT.finditer(text):
        raw, val = m.group(1), float(m.group(1))
        if _in_rule(text, m.start(1), m.end(), policy_texts):
            continue
        if val != int(val):
            bad.append(val)
            errors.append(f"文案金额{raw}元不是整数，必须与工具结果一致")
            continue
        v, lab = int(val), labelled.get(m.start(1))
        if lab:
            label = lab.group(1)
            if label in REFUND_WORDS and "机场建设费" in lab.group(2):
                label = "机场建设费"
            intent, key = LABEL_FIELD[label]
            values = [x for r in fee_res if intent is None or r["intent"] == intent
                      if (x := _num(r["fee"], key)) is not None]
            if not values:
                bad.append(v)
                errors.append(f"文案写了{label}{v}元，但没有对应的工具结果")
            elif v not in values:
                bad.append(v)
                errors.append(f"{label}应为{int(values[0])}元，实际写了{v}元")
        elif v not in shown:
            bad.append(v)
            near = sorted(shown, key=lambda a: abs(a - v))[:1]
            errors.append(f"文案金额{v}元没有工具依据" + (f"，应为{near[0]}元" if near else ""))
    return errors, bad


def _phrase_ok(sentence, m, rule_texts) -> bool:
    phrase = m.group() + CJK_TAIL.match(sentence[m.end():]).group()
    return any(phrase in t for t in rule_texts)


def _unsupported(unit):
    cited, policy = _cited_rules(unit), _cited_rules(unit, only_policy=True)
    rates = {r["fee"].get("rate_pct") for r in _fee_results(unit)}
    norm = [s for s in SPLIT.split(unit["text"]) if s.strip()]
    raw = [s for s in SPLIT.split(unit["raw"]) if s.strip()]
    sentences = raw if len(raw) == len(norm) else norm       # 对外报告原文句子，判断用归一化后的句子
    bad = []
    for s, ns in zip(sentences, norm):
        s_orig, s = s, ns
        for m in CLAIM_KEYS.finditer(s):
            if m.group().endswith("%"):
                ok = int(m.group()[:-1]) in rates or _phrase_ok(s, m, policy)
            else:
                ok = _phrase_ok(s, m, cited)
            if not ok:
                bad.append(s_orig)
                break
    return sentences, bad


def check(scheme: dict, results: list[dict], fact_checker=None) -> dict:
    options, texts = scheme["options"], scheme.get("texts") or []
    if len(texts) == len(options) == len(results) and "".join(texts) == scheme["text"]:
        units = [{"raw": t, "text": _norm(t), "results": [r], "options": [o]}
                 for t, r, o in zip(texts, results, options)]
    else:                       # 拆不开的整段文案（如 LLM 重写）：按全部展示字段的并集校验
        units = [{"raw": scheme["text"], "text": _norm(scheme["text"]), "results": results, "options": options}]
    errors, amount_errors, bad_quotes, sentences, unsupported = [], [], [], [], []
    # 1. 结构化金额逐项比对
    for opt, r in zip(options, results):
        fee = r.get("fee")
        if not fee or not fee.get("allowed") or opt["type"] not in ("CHANGE", "REFUND"):
            continue
        expect_total = fee["total"] if opt["type"] == "CHANGE" else fee["refund_amount"]
        if opt.get("total") != expect_total:
            amount_errors.append(f"{opt['title']} 合计应为{expect_total}元，实际写了{opt.get('total')}元")
        names = {"改期费": fee.get("change_fee"), "票面差价": fee.get("fare_diff"), "退票费": fee.get("refund_fee")}
        for item in opt.get("fee_items", []):
            if item["name"] in names and item["amount"] != names[item["name"]]:
                amount_errors.append(f"{item['name']}应为{names[item['name']]}元，实际写了{item['amount']}元")
    # 2. 文案金额：按子方案、按标签
    for u in units:
        e, b = _check_amounts(u)
        amount_errors += e
        bad_quotes += b
    quote_ok = not amount_errors
    errors += amount_errors
    # 3. 论断核查（短语级）
    for u in units:
        ss, bad = _unsupported(u)
        sentences += ss
        unsupported += [x for x in bad if x not in unsupported]
    if fact_checker is not None:
        ext = fact_checker(sentences, _rule_corpus(results))
        if ext is not None:
            unsupported += [x for x in ext if x in sentences and x not in unsupported]
    for s in unsupported:
        errors.append(f"删除无依据的论断：「{s.strip()}」")
    # 4. 引用必须在本次检索结果里
    retrieved = {x["id"] for r in results for x in r.get("rules", [])}
    bad_refs = [ref for o in options for ref in o.get("rule_refs", []) if ref not in retrieved]
    for ref in bad_refs:
        errors.append(f"引用的规则 {ref} 不在检索结果中")
    # 5. 引擎用到的依据规则必须被检索到并在方案里引用
    rule_missing = False
    for o, r in zip(options, results):
        fee = r.get("fee")
        if fee and not r.get("error") and r["intent"] in ("CHANGE", "REFUND"):
            rid = fee["rule_id"]
            if rid not in {x["id"] for x in r.get("rules", [])}:
                errors.append(f"依据规则 {rid} 未检索到")
                rule_missing = True
            elif rid not in o.get("rule_refs", []):
                errors.append(f"依据规则 {rid} 未在方案中引用")
                rule_missing = True
    # 6. 需要依据的子诉求必须检索到规则
    no_rules = [r["text"] for r in results if r["intent"] != "STATUS" and not r.get("rules")]
    for t in no_rules:
        errors.append(f"「{t}」没有检索到依据规则，不能作答")
    support_ratio = max(0.0, min(1.0, 1 - len(unsupported) / len(sentences))) if sentences else 1.0
    confidence = round(min(1.0 if quote_ok else 0.0, 0.0 if unsupported else 1.0,
                           0.0 if bad_refs or no_rules or rule_missing else 1.0), 2)
    return {"confidence": confidence, "quote_mismatch": not quote_ok, "bad_quotes": bad_quotes,
            "unsupported_claims": unsupported, "bad_refs": bad_refs, "errors": errors,
            "support_ratio": round(support_ratio, 2), "passed": confidence >= _threshold()}


def _threshold():
    from . import settings as S
    return S.get()["verify"]["threshold"]
