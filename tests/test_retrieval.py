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


# ---------- review fixes ----------
import logging
import sys
import threading
import time
import types

import pytest
from air_agent import settings as S

OFF_TOPIC = ["宠物能带上飞机吗", "航班上有wifi吗", "申请发票怎么开", "儿童票怎么买", "行李超重怎么办",
             "飞机餐好吃吗", "明天天气怎么样", "会员积分怎么兑换里程"]
RELEVANT = ["东方云航经济舱折扣票改签收多少钱", "航班取消了怎么退票", "退票手续费", "特价票能改签吗",
            "经济舱全价退票", "可以退吗", "推迟一天要多少钱", "航班取消了怎么办",
            "航班延误了能退票吗", "改签差价怎么算"]


@pytest.mark.parametrize("q", OFF_TOPIC)
def test_off_topic_queries_return_nothing(q):
    assert _search(q)["rules"] == []


@pytest.mark.parametrize("q", RELEVANT)
def test_relevant_queries_hit_something(q):
    assert _search(q)["rules"], q


def test_stopwords_removed_from_bm25_tokens():
    toks = retrieval.tokenize("请问航班取消了怎么办")
    assert not {"航班", "了", "怎么", "请问"} & set(toks)
    assert "取消" in toks


def test_rules_version_bumps_on_edit():
    data.init()
    v0 = data.rules_version()
    r = data.get_rule("GEN-CANCEL")
    data.upsert_rule({k: r[k] for k in ("id", "airline", "cabin", "fare_product", "scenario", "title", "content")})
    v1 = data.rules_version()
    data.delete_rule("GEN-NOSHOW")
    assert v0 < v1 < data.rules_version()


def test_content_edit_reaches_index_without_invalidate():
    data.init()
    retrieval.invalidate()
    retrieval.search("退票手续费", {})
    r = data.get_rule("MU-Y-RFD")
    data.upsert_rule({**{k: r[k] for k in ("id", "airline", "cabin", "fare_product", "scenario", "title")},
                      "content": r["content"] + "退票须通过星河通道办理。"})
    out = retrieval.search("退票星河通道", {})
    assert any(x["id"] == "MU-Y-RFD" and "星河通道" in x["content"] for x in out["rules"])


def test_deleted_rule_leaves_index_without_invalidate():
    data.init()
    retrieval.invalidate()
    assert any(x["id"] == "GEN-CANCEL" for x in retrieval.search("航班取消了怎么退票", {})["rules"])
    data.delete_rule("GEN-CANCEL")
    assert all(x["id"] != "GEN-CANCEL" for x in retrieval.search("航班取消了怎么退票", {})["rules"])


def test_empty_rule_set_returns_empty_funnel():
    data.init()
    for r in data.list_rules():
        data.delete_rule(r["id"])
    out = retrieval.search("退票", {})
    assert out["rules"] == []
    assert [s["stage"] for s in out["funnel"]] == STAGES
    assert all(s["count"] == 0 for s in out["funnel"])


def test_load_models_runs_once_under_concurrency(monkeypatch):
    calls = {"n": 0}

    class FakeEmbed:
        def __init__(self, *a, **k):
            calls["n"] += 1
            time.sleep(0.2)

    class FakeRanker:
        def __init__(self, *a, **k):
            pass

    monkeypatch.setitem(sys.modules, "fastembed", types.SimpleNamespace(TextEmbedding=FakeEmbed))
    monkeypatch.setitem(sys.modules, "flashrank", types.SimpleNamespace(Ranker=FakeRanker))
    monkeypatch.setattr(retrieval, "_models", {"loaded": False, "embed": None, "rerank": None, "error": ""})
    seen = []

    def work():
        retrieval._load_models()
        seen.append(retrieval._models["embed"] is not None)

    ts = [threading.Thread(target=work) for _ in range(6)]
    [t.start() for t in ts]
    [t.join() for t in ts]
    assert calls["n"] == 1 and all(seen)


def test_load_failure_is_recorded(monkeypatch):
    monkeypatch.setitem(sys.modules, "fastembed", None)   # import 失败
    monkeypatch.setattr(retrieval, "_models", {"loaded": False, "embed": None, "rerank": None, "error": ""})
    retrieval._load_models()
    assert retrieval._models["loaded"] and retrieval._models["error"]


def test_runtime_model_failure_degrades_to_fallback(monkeypatch):
    class Boom:
        def embed(self, texts):
            raise RuntimeError("onnx exploded")

        query_embed = embed

    S.update({"retrieval": {"backend": "hybrid"}})
    monkeypatch.setattr(retrieval, "_models", {"loaded": True, "embed": Boom(), "rerank": object(), "error": ""})
    data.init()
    retrieval.invalidate()
    out = retrieval.search("航班取消了怎么退票", {})
    assert out["backend"] == "fallback" and out["degraded"] is True
    assert "onnx exploded" in out["degraded_reason"]
    assert out["rules"][0]["id"] == "GEN-CANCEL"


def test_boost_cannot_lift_candidate_over_rerank_min():
    data.init()
    retrieval.invalidate()
    f = {"fare_product": "Y", "scenario": "VOLUNTARY_REFUND"}
    S.update({"retrieval": {"rerank_min": 0.0}})
    plain = retrieval.search("经济舱全价退票", f)
    ranked = next(s for s in plain["funnel"] if s["stage"] == "rerank")["items"]
    hu = next(i for i in ranked if i["id"].startswith("HU-"))
    assert 0 < hu["score"] < 1
    S.update({"retrieval": {"rerank_min": round(hu["score"] + 0.001, 4)}})
    out = retrieval.search("经济舱全价退票", f, {"airlines": ["HU"]})
    assert all(not x["id"].startswith("HU-") for x in out["rules"])


def test_jieba_logger_silenced_at_import():
    import jieba
    assert jieba.default_logger.level == logging.WARNING
