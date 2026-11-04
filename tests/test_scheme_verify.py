from air_agent import data, engine, retrieval, scheme, verify

D = "20260926"


def _results():
    data.init()
    retrieval.invalidate()
    o = data.get_order(f"ORD{D}001")
    chg = engine.calc(o["order_id"], "CHANGE", target_date="2026-09-29")
    rfd = engine.calc(o["order_id"], "REFUND")
    r1 = retrieval.search("经济舱折扣改签", {"airline": "MU", "fare_product": "Y_DISCOUNT",
                                         "scenario": "VOLUNTARY_CHANGE"})["rules"]
    r2 = retrieval.search("经济舱折扣退票", {"airline": "MU", "fare_product": "Y_DISCOUNT",
                                         "scenario": "VOLUNTARY_REFUND"})["rules"]
    return [{"intent": "CHANGE", "text": "改到大后天", "target_date": "2026-09-29", "status": None,
             "rules": r1, "fee": chg, "error": None},
            {"intent": "REFUND", "text": "退票多少钱", "target_date": None, "status": None,
             "rules": r2, "fee": rfd, "error": None}]


def test_template_scheme_passes_verify():
    res = _results()
    s = scheme.generate(res)
    assert [o.type for o in s.options] == ["CHANGE", "REFUND"]
    assert s.options[0].total == res[0]["fee"]["total"]
    v = verify.check(s.model_dump(), res)
    assert v["confidence"] == 1.0 and v["passed"]


def test_wrong_amount_detected():
    res = _results()
    s = scheme.generate(res).model_dump()
    wrong = res[0]["fee"]["change_fee"] + 20
    s["text"] = s["text"].replace(f"改期费{res[0]['fee']['change_fee']}元", f"改期费{wrong}元")
    v = verify.check(s, res)
    assert not v["passed"] and v["quote_mismatch"]
    assert any(str(res[0]["fee"]["change_fee"]) in e for e in v["errors"])


def test_unsupported_claim_detected():
    res = _results()
    s = scheme.generate(res).model_dump()
    s["text"] += "另外，金卡会员可以免费改签3次。"
    v = verify.check(s, res)
    assert not v["passed"] and v["unsupported_claims"]


def test_bad_rule_ref_detected():
    res = _results()
    s = scheme.generate(res).model_dump()
    s["options"][0]["rule_refs"] = ["CA-F-CHG"]
    v = verify.check(s, res)
    assert not v["passed"] and v["bad_refs"] == ["CA-F-CHG"]


def test_feedback_lists_concrete_fixes():
    res = _results()
    s = scheme.generate(res).model_dump()
    s["options"][0]["total"] = 1
    v = verify.check(s, res)
    assert any("应为" in e for e in v["errors"])


def test_not_allowed_option():
    data.init()
    sp = next(o for o in data.list_orders() if o["fare_product"] == "SPECIAL")
    fee = engine.calc(sp["order_id"], "CHANGE", target_date="2026-10-02")
    rules = [data.get_rule(fee["rule_id"])]
    res = [{"intent": "CHANGE", "text": "改签", "target_date": "2026-10-02", "status": None,
            "rules": rules, "fee": fee, "error": None}]
    s = scheme.generate(res)
    assert s.options[0].allowed is False and "不得自愿改签" in s.text
    assert verify.check(s.model_dump(), res)["passed"]


def _rules_for(order, scenario, query):
    return retrieval.search(query, {"airline": order["airline"], "cabin": order["cabin"],
                                    "fare_product": order["fare_product"], "scenario": scenario})["rules"]


def test_cancelled_flight_refund_passes_verify():
    data.init()
    retrieval.invalidate()
    o = data.get_order(f"ORD{D}007")
    fee = engine.calc(o["order_id"], "REFUND")
    assert fee["involuntary"] and fee["rule_id"] == "GEN-CANCEL"
    rules = _rules_for(o, "CANCEL", "航班取消退票")
    assert any(r["id"] == "GEN-CANCEL" for r in rules)
    res = [{"intent": "REFUND", "text": "航班取消了退票", "target_date": None, "status": None,
            "rules": rules, "fee": fee, "error": None}]
    s = scheme.generate(res)
    assert s.options[0].total == fee["refund_amount"] and s.options[0].rule_refs == ["GEN-CANCEL"]
    v = verify.check(s.model_dump(), res)
    assert v["passed"] and v["confidence"] == 1.0, (v, s.text)


def test_delayed_flight_change_passes_verify():
    data.init()
    retrieval.invalidate()
    o = data.get_order(f"ORD{D}008")
    fee = engine.calc(o["order_id"], "CHANGE", target_date="2026-09-30")
    assert fee["involuntary"] and fee["rule_id"] == "GEN-DELAY"
    rules = _rules_for(o, "DELAY", "航班延误改签")
    assert any(r["id"] == "GEN-DELAY" for r in rules)
    res = [{"intent": "CHANGE", "text": "航班延误想改签", "target_date": "2026-09-30", "status": None,
            "rules": rules, "fee": fee, "error": None}]
    s = scheme.generate(res)
    assert s.options[0].total == 0 and s.options[0].rule_refs == ["GEN-DELAY"]
    v = verify.check(s.model_dump(), res)
    assert v["passed"] and v["confidence"] == 1.0, (v, s.text)


def test_cancelled_flight_change_and_delayed_refund_pass_verify():
    data.init()
    retrieval.invalidate()
    o7, o8 = data.get_order(f"ORD{D}007"), data.get_order(f"ORD{D}008")
    c7 = engine.calc(o7["order_id"], "CHANGE", target_date="2026-09-28", target_flight=None)
    r8 = engine.calc(o8["order_id"], "REFUND")
    res = [{"intent": "CHANGE", "text": "改签", "target_date": "2026-09-28", "status": None,
            "rules": _rules_for(o7, "CANCEL", "航班取消改签"), "fee": c7, "error": None},
           {"intent": "REFUND", "text": "退票", "target_date": None, "status": None,
            "rules": _rules_for(o8, "DELAY", "航班延误退票"), "fee": r8, "error": None}]
    s = scheme.generate(res)
    v = verify.check(s.model_dump(), res)
    assert v["passed"], (v, s.text)


def test_not_allowed_refund_options_pass_verify():
    data.init()
    for prod, al in (("GROUP", None), ):
        o = next(x for x in data.list_orders() if x["fare_product"] == prod)
        fee = engine.calc(o["order_id"], "REFUND")
        assert fee["allowed"] is False
        res = [{"intent": "REFUND", "text": "退票", "target_date": None, "status": None,
                "rules": [data.get_rule(fee["rule_id"])], "fee": fee, "error": None}]
        s = scheme.generate(res)
        assert s.options[0].allowed is False
        v = verify.check(s.model_dump(), res)
        assert v["passed"], (v, s.text)


def test_empty_rules_fails_verify():
    res = _results()
    res[0]["rules"] = []
    s = scheme.generate(res).model_dump()
    v = verify.check(s, res)
    assert not v["passed"] and any("没有检索到依据规则" in e for e in v["errors"])


def test_every_seed_rule_as_policy_answer_passes_verify():
    """POLICY 方案原样引用规则原文（含「机场建设费50元」之类的固定金额），不应被金额校验误杀。"""
    data.init()
    for r in data.list_rules():
        res = [{"intent": "POLICY", "text": "政策咨询", "target_date": None, "status": None,
                "rules": [r], "fee": None, "error": None}]
        s = scheme.generate(res)
        v = verify.check(s.model_dump(), res)
        assert v["passed"], (r["id"], v["errors"])


def test_rule_amount_exemption_does_not_hide_invented_amounts():
    data.init()
    r = data.get_rule("MU-Y_DISCOUNT-RFD")
    res = [{"intent": "POLICY", "text": "政策咨询", "target_date": None, "status": None,
            "rules": [r], "fee": None, "error": None}]
    s = scheme.generate(res).model_dump()
    s["text"] += "另需支付777元。"
    v = verify.check(s, res)
    assert not v["passed"] and v["bad_quotes"] == [777]


# ---------------------------------------------------------------- 加固：金额逐项/逐标签校验
import pytest
from datetime import date, timedelta
from air_agent import seed
from air_agent import settings as S
from air_agent.engine import EngineError


def _edit(s, idx, old, new, split):
    """改文案。split=True 时同步改 texts（按子方案校验）；False 时只改 text（模拟 LLM 重写、无法拆分）。"""
    assert old in s["texts"][idx], (old, s["texts"][idx])
    s["text"] = s["text"].replace(old, new)
    if split:
        s["texts"][idx] = s["texts"][idx].replace(old, new)
    return s


def _wrong(res):
    return res[0]["fee"]["change_fee"] + 20


def _amount_cases(res):
    chg, rfd = res[0]["fee"], res[1]["fee"]
    cf, fd, ra, rf = chg["change_fee"], chg["fare_diff"], rfd["refund_amount"], rfd["refund_fee"]
    w = _wrong(res)
    assert cf != fd and cf != 50 and rf != ra
    return {
        "swap_fee_and_diff": (0, f"改期费{cf}元，票面差价{fd}元", f"改期费{fd}元，票面差价{cf}元"),
        "refund_equals_fare": (1, f"预计退款{ra}元", f"预计退款{rfd['fare']}元"),
        "refund_equals_fee": (1, f"预计退款{ra}元", f"预计退款{rf}元"),
        "fee_equals_tax": (0, f"改期费{cf}元", "改期费50元"),
        "fullwidth_yen": (0, f"改期费{cf}元", f"改期费￥{w}"),
        "rmb": (0, f"改期费{cf}元", f"改期费RMB {w}"),
        "kuai": (0, f"改期费{cf}元", f"改期费{w}块"),
        "chinese_numeral": (0, f"改期费{cf}元", "改期费五百八十元"),
        "thousands_comma": (0, f"改期费{cf}元", f"改期费1,{cf}元"),
        "fraction": (0, f"改期费{cf}元", f"改期费{cf}.90元"),
    }


@pytest.mark.parametrize("split", [True, False])
@pytest.mark.parametrize("case", list(_amount_cases([
    {"fee": {"change_fee": 291, "fare_diff": 0, "refund_amount": 632, "refund_fee": 388, "fare": 970}},
    {"fee": {"refund_amount": 632, "refund_fee": 388, "fare": 970}}])))
def test_tampered_amounts_are_rejected(case, split):
    res = _results()
    idx, old, new = _amount_cases(res)[case]
    s = _edit(scheme.generate(res).model_dump(), idx, old, new, split)
    v = verify.check(s, res)
    assert not v["passed"] and v["quote_mismatch"], (case, s["text"], v)


@pytest.mark.parametrize("split", [True, False])
def test_correct_label_with_yen_passes(split):
    res = _results()
    cf = res[0]["fee"]["change_fee"]
    s = _edit(scheme.generate(res).model_dump(), 0, f"改期费{cf}元", f"改期费￥{cf}", split)
    assert verify.check(s, res)["passed"]


def test_amount_from_other_option_rejected_per_option():
    res = _results()
    s = scheme.generate(res).model_dump()
    ra = res[1]["fee"]["refund_amount"]
    s["texts"][0] += f"预计金额{ra}元。"
    s["text"] = "".join(s["texts"])
    v = verify.check(s, res)
    assert not v["passed"] and ra in v["bad_quotes"]
    # 拆不开的整段文案只能按全部展示字段并集校验
    s2 = scheme.generate(res).model_dump()
    s2["text"] += f"预计金额{ra}元。"
    assert verify.check(s2, res)["passed"]


def test_generate_keeps_per_option_texts_and_enforce_keeps_template_texts():
    res = _results()
    out = scheme.generate(res)
    assert len(out.texts) == 2 and "".join(out.texts) == out.text

    class Llm:
        def write_scheme(self, out, results, feedback, context):
            return out.model_copy(update={"text": "润色后的整段文案"})
    out2 = scheme.generate(res, llm=Llm())
    assert out2.text == "润色后的整段文案" and out2.texts == out.texts


# ---------------------------------------------------------------- 加固：论断
def _policy_results():
    data.init()
    rs = [data.get_rule("MU-Y_DISCOUNT-CHG"), data.get_rule("MU-Y_DISCOUNT-RFD")]
    return [{"intent": "POLICY", "text": "政策", "target_date": None, "status": None, "rules": [r],
             "fee": None, "error": None} for r in rs]


def test_single_unsupported_sentence_is_hard_fail_even_in_long_text():
    res = _policy_results()
    s = scheme.generate(res).model_dump()
    s["text"] += "另外，金卡会员可以免费改签3次。"
    v = verify.check(s, res)
    assert v["support_ratio"] >= 0.85          # 旧逻辑下比例够高会放行
    assert v["unsupported_claims"] and v["confidence"] == 0.0 and not v["passed"]


@pytest.mark.parametrize("claim", ["无需支付任何费用。", "不收手续费。", "保证改签成功。", "航班可随便改。",
                                   "打折票也一样。", "不限次数。", "承诺赔付。", "百分之百退款。"])
def test_extended_claim_keywords_are_detected(claim):
    res = _results()
    s = scheme.generate(res).model_dump()
    s["text"] += claim
    v = verify.check(s, res)
    assert not v["passed"] and v["unsupported_claims"], claim


def test_phrase_level_support_full_refund_claim():
    res = _results()
    s = scheme.generate(res).model_dump()
    s["text"] += "可全额退款。"
    v = verify.check(s, res)
    assert not v["passed"] and v["unsupported_claims"]


def test_percent_claim_must_match_engine_rate():
    res = _results()
    rate = res[1]["fee"]["rate_pct"]
    bad = scheme.generate(res).model_dump()
    bad["text"] += f"退票只收{rate + 5}%手续费。"
    assert not verify.check(bad, res)["passed"]
    ok = scheme.generate(res).model_dump()
    ok["text"] += f"退票只收{rate}%手续费。"
    assert verify.check(ok, res)["passed"]


def test_gen_cancel_full_refund_wording_passes():
    data.init()
    retrieval.invalidate()
    o = data.get_order(f"ORD{D}007")
    fee = engine.calc(o["order_id"], "REFUND")
    res = [scheme.ensure_fee_rule({"intent": "REFUND", "text": "退票", "target_date": None, "status": None,
                                   "rules": [], "fee": fee, "error": None})]
    s = scheme.generate(res).model_dump()
    s["text"] += "因航班取消，可办理全额退票。"
    assert verify.check(s, res)["passed"]
    s["text"] += "可办理全额退款。"
    assert not verify.check(s, res)["passed"]


# ---------------------------------------------------------------- 加固：引擎依据规则
def test_fee_rule_must_be_retrieved_and_cited():
    res = _results()
    res[0]["rules"] = [data.get_rule("GEN-DELAY")]
    s = scheme.generate(res).model_dump()
    v = verify.check(s, res)
    assert not v["passed"] and any("依据规则 MU-Y_DISCOUNT-CHG 未检索到" in e for e in v["errors"])


def test_ensure_fee_rule_injects_without_mutating():
    res = _results()
    r = {**res[0], "rules": [data.get_rule("GEN-DELAY")]}
    fixed = scheme.ensure_fee_rule(r)
    assert [x["id"] for x in fixed["rules"]][0] == "MU-Y_DISCOUNT-CHG" and len(r["rules"]) == 1
    assert scheme.ensure_fee_rule(fixed)["rules"] == fixed["rules"]
    pol = {"intent": "POLICY", "rules": [], "fee": None}
    assert scheme.ensure_fee_rule(pol) is pol or scheme.ensure_fee_rule(pol) == pol
    res2 = [fixed, res[1]]
    assert verify.check(scheme.generate(res2).model_dump(), res2)["passed"]


def test_option_must_cite_engine_rule():
    res = _results()
    s = scheme.generate(res).model_dump()
    s["options"][0]["rule_refs"] = []
    v = verify.check(s, res)
    assert not v["passed"] and any("MU-Y_DISCOUNT-CHG" in e for e in v["errors"])


# ---------------------------------------------------------------- 加固：fact_checker 合并
def test_fact_checker_is_unioned_with_keyword_result():
    res = _results()
    s = scheme.generate(res).model_dump()
    s["text"] += "另外金卡会员可以免费改签3次。"
    assert not verify.check(s, res, fact_checker=lambda sents, corpus: [])["passed"]
    clean = scheme.generate(res).model_dump()
    first = [x for x in clean["text"].replace("；", "。").split("。") if x][0]
    v = verify.check(clean, res, fact_checker=lambda sents, corpus: [first, "不存在的句子"])
    assert v["unsupported_claims"] == [first] and not v["passed"]
    assert 0.0 <= v["support_ratio"] <= 1.0
    v2 = verify.check(clean, res, fact_checker=lambda sents, corpus: sents * 3)
    assert 0.0 <= v2["support_ratio"] <= 1.0


# ---------------------------------------------------------------- 加固：全量扫描，锁定“无误杀”
def _rules_via_search(o, intent, fee, hint):
    scen = {"GEN-CANCEL": "CANCEL", "GEN-DELAY": "DELAY"}.get(
        fee["rule_id"], "VOLUNTARY_CHANGE" if intent == "CHANGE" else "VOLUNTARY_REFUND")
    q = seed.PRODUCTS[o["fare_product"]][1] + hint
    return retrieval.search(q, {"airline": o["airline"], "cabin": o["cabin"],
                                "fare_product": o["fare_product"], "scenario": scen})["rules"]


def _targets(o):
    base = max(date.fromisoformat(o["date"]), S.now().date())
    for d in range(1, 4):
        day = (base + timedelta(days=d)).isoformat()
        if data.get_flight(o["flight_no"], day):
            yield day


@pytest.mark.parametrize("n", range(1, 13))
def test_sweep_templates_pass_verify_for_every_order(n):
    data.init()
    retrieval.invalidate()
    o = data.get_order(f"ORD{D}{n:03d}")
    cases = []
    for day in _targets(o):
        try:
            fee = engine.calc(o["order_id"], "CHANGE", target_date=day)
        except EngineError:
            continue
        cases.append(("CHANGE", day, fee, "改签规则"))
    try:
        cases.append(("REFUND", None, engine.calc(o["order_id"], "REFUND"), "退票规则"))
    except EngineError:
        pass
    assert cases
    for intent, day, fee, hint in cases:
        res = [scheme.ensure_fee_rule({"intent": intent, "text": hint, "target_date": day, "status": None,
                                       "rules": _rules_via_search(o, intent, fee, hint), "fee": fee,
                                       "error": None})]
        s = scheme.generate(res)
        v = verify.check(s.model_dump(), res)
        assert v["passed"] and v["confidence"] == 1.0, (o["order_id"], intent, day, s.text, v)
    # 同一订单改签 + 退票合并成多诉求方案
    if len(cases) >= 2:
        res = [scheme.ensure_fee_rule({"intent": i, "text": h, "target_date": d, "status": None,
                                       "rules": _rules_via_search(o, i, f, h), "fee": f, "error": None})
               for i, d, f, h in [(c[0], c[1], c[2], c[3]) for c in (cases[0], cases[-1])]]
        s = scheme.generate(res)
        v = verify.check(s.model_dump(), res)
        assert v["passed"] and v["confidence"] == 1.0, (o["order_id"], s.text, v)
