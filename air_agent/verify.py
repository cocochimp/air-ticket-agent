"""工作五：金额后验、论断事实核查、引用合法性，计算置信度。"""
import re

AMOUNT = re.compile(r"(\d+(?:\.\d+)?)\s*元")
CLAIM_KEYS = re.compile(r"免费|免收|全额|不得|不可|可以|\d+次|\d+%|会员|积分|赔偿|补偿")


def _allowed_amounts(results):
    nums = set()
    for r in results:
        f = r.get("fee") or {}
        for k in ("change_fee", "fare_diff", "total", "refund_fee", "refund_amount", "tax", "fare"):
            if isinstance(f.get(k), (int, float)):
                nums.add(int(f[k]))
    return nums


def _rule_corpus(results):
    return "".join(f"{x['title']}{x['content']}" for r in results for x in r.get("rules", []))


def _supported(sentence: str, corpus: str) -> bool:
    for kw in CLAIM_KEYS.findall(sentence):
        if kw not in corpus:
            return False
    return True


def check(scheme: dict, results: list[dict], fact_checker=None) -> dict:
    errors = []
    allowed = _allowed_amounts(results)
    # 规则原文自带的固定金额（如「机场建设费50元」）可以原样引用；其余金额必须来自引擎
    rule_amounts = {int(float(x)) for x in AMOUNT.findall(_rule_corpus(results))}
    # 1. 文案金额必须来自引擎
    quoted = [int(float(x)) for x in AMOUNT.findall(scheme["text"])]
    bad_quotes = [q for q in quoted if q not in allowed and q not in rule_amounts]
    # 2. 结构化金额逐项比对
    for opt, r in zip(scheme["options"], results):
        fee = r.get("fee")
        if not fee or not fee.get("allowed") or opt["type"] not in ("CHANGE", "REFUND"):
            continue
        expect_total = fee["total"] if opt["type"] == "CHANGE" else fee["refund_amount"]
        if opt.get("total") != expect_total:
            errors.append(f"{opt['title']} 合计应为{expect_total}元，实际写了{opt.get('total')}元")
        names = {"改期费": fee.get("change_fee"), "票面差价": fee.get("fare_diff"), "退票费": fee.get("refund_fee")}
        for item in opt.get("fee_items", []):
            if item["name"] in names and item["amount"] != names[item["name"]]:
                errors.append(f"{item['name']}应为{names[item['name']]}元，实际写了{item['amount']}元")
    for q in bad_quotes:
        near = sorted(allowed, key=lambda a: abs(a - q))[:1]
        errors.append(f"文案金额{q}元没有工具依据" + (f"，应为{near[0]}元" if near else ""))
    quote_ok = not bad_quotes and not errors
    # 3. 论断事实核查
    corpus = _rule_corpus(results)
    sentences = [s for s in re.split(r"[。！？；]", scheme["text"]) if s.strip()]
    unsupported = [s for s in sentences if not _supported(s, corpus)]
    if fact_checker is not None:
        unsupported = fact_checker(sentences, corpus) or unsupported
    for s in unsupported:
        errors.append(f"删除无依据的论断：「{s.strip()}」")
    # 4. 引用必须在本次检索结果里
    retrieved = {x["id"] for r in results for x in r.get("rules", [])}
    bad_refs = [ref for o in scheme["options"] for ref in o.get("rule_refs", []) if ref not in retrieved]
    for ref in bad_refs:
        errors.append(f"引用的规则 {ref} 不在检索结果中")
    # 5. 需要依据的子诉求必须检索到规则
    no_rules = [r["text"] for r in results if r["intent"] != "STATUS" and not r.get("rules")]
    for t in no_rules:
        errors.append(f"「{t}」没有检索到依据规则，不能作答")
    support_ratio = 1 - len(unsupported) / len(sentences) if sentences else 1.0
    confidence = round(min(1.0 if quote_ok else 0.0, support_ratio, 0.0 if bad_refs or no_rules else 1.0), 2)
    return {"confidence": confidence, "quote_mismatch": not quote_ok, "bad_quotes": bad_quotes,
            "unsupported_claims": unsupported, "bad_refs": bad_refs, "errors": errors,
            "support_ratio": round(support_ratio, 2), "passed": confidence >= _threshold()}


def _threshold():
    from . import settings as S
    return S.get()["verify"]["threshold"]
