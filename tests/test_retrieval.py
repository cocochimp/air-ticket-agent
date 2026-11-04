from air_agent import data, retrieval

STAGES = ["dense", "sparse", "rrf", "filter", "rerank", "mmr", "inject"]


def _search(q, **filters):
    data.init()
    retrieval.invalidate()
    return retrieval.search(q, filters)


def test_voluntary_change_hits_product_rule():
    r = _search("东方云航经济舱折扣票改签收多少钱", airline="MU", fare_product="Y_DISCOUNT",
                scenario="VOLUNTARY_CHANGE")
    assert r["rules"][0]["id"] == "MU-Y_DISCOUNT-CHG"
    assert [s["stage"] for s in r["funnel"]] == STAGES
    assert r["backend"] == "fallback"


def test_cancel_scenario_prefers_generic_rule():
    r = _search("航班取消了怎么退票", airline="MU", fare_product="Y", scenario="CANCEL")
    assert r["rules"][0]["id"] == "GEN-CANCEL"


def test_hard_filter_removes_other_airlines():
    r = _search("退票手续费", airline="CA", fare_product="C", scenario="VOLUNTARY_REFUND")
    kept = next(s for s in r["funnel"] if s["stage"] == "filter")["items"]
    assert all(i["id"].startswith(("CA-C-", "GEN-")) for i in kept)


def test_funnel_counts_shrink_after_fusion():
    r = _search("特价票能改签吗")
    counts = {s["stage"]: s["count"] for s in r["funnel"]}
    assert counts["rrf"] <= 40 and counts["rerank"] <= 10 and counts["mmr"] <= 5 and counts["inject"] <= 3
    assert counts["filter"] >= counts["rerank"] >= counts["mmr"] >= counts["inject"]
    assert any(x["fare_product"] == "SPECIAL" for x in r["rules"])


def test_profile_boost_changes_score():
    data.init()
    retrieval.invalidate()
    plain = retrieval.search("经济舱全价退票", {"fare_product": "Y", "scenario": "VOLUNTARY_REFUND"})
    boosted = retrieval.search("经济舱全价退票", {"fare_product": "Y", "scenario": "VOLUNTARY_REFUND"},
                               {"airlines": ["HU"]})
    assert boosted["rules"][0]["airline"] == "HU"
    assert plain["rules"][0]["score"] <= boosted["rules"][0]["score"]


def test_off_topic_returns_nothing():
    r = _search("宠物能带上飞机吗")
    assert r["rules"] == []


def test_index_rebuilds_after_rule_edit():
    data.init()
    retrieval.invalidate()
    retrieval.search("宠物托运", {})
    data.upsert_rule({"id": "GEN-PET", "airline": "*", "cabin": "*", "fare_product": "*", "scenario": "FARE_DIFF",
                      "title": "宠物托运改签说明", "content": "携带宠物托运的旅客改签需重新申请宠物舱位。"})
    r = retrieval.search("宠物托运改签", {})
    assert any(x["id"] == "GEN-PET" for x in r["rules"])
