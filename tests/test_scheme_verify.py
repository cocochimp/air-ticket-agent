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
