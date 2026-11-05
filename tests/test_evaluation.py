import json
from air_agent import evaluation as E


def test_build_cases_cover_spec():
    cases = E.build_cases()
    assert len(cases) >= 60
    assert len({c["id"] for c in cases}) == len(cases)
    tags = {t for c in cases for t in c["tags"]}
    assert {"单诉求", "多诉求", "政策咨询", "追问", "指代", "转人工", "非自愿", "不可办理"} <= tags
    assert {c["airline"] for c in cases if c["airline"]} == {"MU", "CA", "CZ", "HU"}


def test_built_cases_validate_clean():
    cases = E.build_cases()
    assert E.validate_cases(cases) == {}


def test_validation_reports_field_errors():
    c = E.build_cases()[0]
    bad = json.loads(json.dumps(c))
    bad["id"] = "bad-1"
    bad["order"] = "#99"
    bad["expect"]["tools"] = ["tool_magic"]
    bad["expect"]["rule_ids"] = ["NOPE"]
    errs = E.validate_cases([bad])["bad-1"]
    fields = {e["field"] for e in errs}
    assert {"order", "expect.tools", "expect.rule_ids"} <= fields


def test_validation_catches_fee_mismatch():
    c = next(x for x in E.build_cases() if x["expect"]["fees"].get("CHANGE"))
    c["expect"]["fees"]["CHANGE"] += 1
    errs = E.validate_cases([c])[c["id"]]
    assert any(e["field"] == "expect.fees.CHANGE" and "引擎值" in e["message"] for e in errs)


def test_duplicate_ids_rejected():
    c = E.build_cases()[0]
    errs = E.validate_cases([c, c])
    assert any(e["field"] == "id" for e in errs[c["id"]])


def test_run_subset_and_summary():
    cases = [c for c in E.build_cases() if "多诉求" in c["tags"]][:3]
    run = E.run_cases(cases, mode="offline")
    s = run["summary"]
    assert s["n"] == 3
    for k in ("tool_acc", "policy_hit", "correctness", "human_rate", "latency_avg_ms", "latency_p95_ms",
              "tokens_avg", "failure_reasons", "by_group"):
        assert k in s
    assert run["results"][0]["actual"]["tools"]


def test_full_gold_offline_baseline():
    run = E.run_cases(E.build_cases(), mode="offline")
    s = run["summary"]
    failed = [(r["case_id"], r["reasons"]) for r in run["results"] if not r["passed"]]
    assert s["correctness"] >= 0.9, failed
    assert s["tool_acc"] >= 0.9, failed
