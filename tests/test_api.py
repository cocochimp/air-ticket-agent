"""Web 接口测试：TestClient + 假的 graph / 评测命令行，保持快速。"""
import json
import sys
import types

import pytest
from fastapi.testclient import TestClient

import app as web
from air_agent import data


def parse_sse(text: str) -> list[tuple[str, dict]]:
    out = []
    for block in text.strip().split("\n\n"):
        ev, payload = None, None
        for line in block.splitlines():
            if line.startswith("event: "):
                ev = line[7:]
            elif line.startswith("data: "):
                payload = json.loads(line[6:])
        if ev:
            out.append((ev, payload))
    return out


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(web, "EVAL_CASES", tmp_path / "eval" / "gold_cases.json")
    with TestClient(web.app) as c:
        yield c


@pytest.fixture
def fake_graph(monkeypatch):
    """注入一个假的 air_agent.graph 模块，记录调用参数。"""
    calls = {}
    mod = types.ModuleType("air_agent.graph")

    def result(session_id, outcome="ANSWER"):
        return {"session_id": session_id, "outcome": outcome,
                "final": {"outcome": outcome, "text": "改签费 90 元", "options": []},
                "interrupt": {"kind": "confirm"} if outcome == "ANSWER" else None, "trace": [],
                "parsed": {}, "context": {}, "results": [], "verify": {}, "memory": {}}

    def run_turn(session_id, user_id, message, order_id=None, mode=None, client=None, on_event=None):
        calls["run_turn"] = dict(session_id=session_id, user_id=user_id, message=message, order_id=order_id, mode=mode)
        on_event({"type": "trace", "event": {"node": "user_input", "status": "ok", "ms": 1}})
        on_event({"type": "trace", "event": {"node": "tool_rule_retrieve", "status": "ok", "ms": 2}})
        on_event({"type": "interrupt", "payload": {"kind": "confirm", "order_id": order_id, "options": []}})
        return result(session_id)

    def resume(session_id, payload, client=None, on_event=None):
        calls["resume"] = dict(session_id=session_id, payload=payload)
        on_event({"type": "trace", "event": {"node": "tool_issue_ticket", "status": "ok", "ms": 3}})
        return result(session_id, "TICKETED")

    mod.run_turn, mod.resume = run_turn, resume
    mod.pending_interrupt = lambda sid: {"kind": "confirm"} if sid == "s1" else None
    monkeypatch.setitem(sys.modules, "air_agent.graph", mod)
    import air_agent
    monkeypatch.setattr(air_agent, "graph", mod, raising=False)
    return calls


def first_order(client, user="u_zhang"):
    return client.get(f"/api/orders?user_id={user}").json()["orders"][0]


# ---------------- 配置 ----------------
def test_config_get_put_and_errors(client):
    cfg = client.get("/api/config").json()
    assert cfg["retrieval"]["k_recall"] == 60
    r = client.put("/api/config", json={"retrieval": {"k_inject": 4}, "verify": {"threshold": 0.9}})
    assert r.status_code == 200 and r.json()["retrieval"]["k_inject"] == 4
    assert client.get("/api/config").json()["verify"]["threshold"] == 0.9

    r = client.put("/api/config", json={"verify": {"threshold": 1.5}})
    assert r.status_code == 400 and "verify.threshold" in r.json()["error"]["fields"]
    r = client.put("/api/config", json={"retrieval": {"k_inject": "x"}})
    assert r.status_code == 400 and r.json()["error"]["code"] == "VALIDATION"
    r = client.put("/api/config", json={"retrieval": {"nope": 1}})
    assert r.status_code == 400
    r = client.put("/api/config", json={"bogus": 1})
    assert r.status_code == 422 and "bogus" in r.json()["error"]["fields"]


def test_config_masked_key_and_reset(client):
    client.put("/api/config", json={"llm": {"api_key": "sk-1234567890abcdef"}, "verify": {"max_retries": 2}})
    masked = client.get("/api/config").json()["llm"]["api_key"]
    assert "****" in masked and "567890" not in masked
    client.put("/api/config", json={"llm": {"api_key": masked}})      # 打码值不覆盖真实 key
    assert client.get("/api/status").json()["deepseek_configured"] is True
    cfg = client.post("/api/config/reset").json()
    assert cfg["verify"]["max_retries"] == 1 and "****" in cfg["llm"]["api_key"]


# ---------------- 演示数据 ----------------
def test_orders_users_and_flight_status(client):
    orders = client.get("/api/orders?user_id=u_zhang").json()["orders"]
    assert orders and all(o["user_id"] == "u_zhang" for o in orders)
    o = orders[0]
    assert o["flight"]["origin_city"] and o["product_label"] and o["flight_status_label"]
    users = client.get("/api/users").json()["users"]
    assert {u["user_id"] for u in users} == {"u_zhang", "u_li", "u_wang"}
    assert not any(k.startswith("_") for u in users for k in u["data"])

    r = client.post(f"/api/flights/{o['flight_no']}/{o['date']}/status", json={"status": "delayed", "delay_min": 150})
    assert r.status_code == 200 and r.json()["flight"]["delay_min"] == 150
    r = client.post(f"/api/flights/{o['flight_no']}/{o['date']}/status", json={"status": "boom"})
    assert r.status_code == 422
    assert client.get("/api/orders?user_id=bad id").status_code == 400


# ---------------- 规则 ----------------
RULE = {"id": "TEST-PET", "airline": "*", "cabin": "*", "fare_product": "*", "scenario": "CANCEL",
        "title": "测试规则：宠物托运", "content": "宠物托运需提前申请，退票时宠物运费全额退还。"}


def test_rules_crud_and_search_debug(client):
    n = len(client.get("/api/rules").json()["rules"])
    assert client.post("/api/rules", json=RULE).status_code == 201
    assert client.post("/api/rules", json=RULE).status_code == 409
    assert len(client.get("/api/rules").json()["rules"]) == n + 1
    r = client.put("/api/rules/TEST-PET", json={**RULE, "title": "改过的标题"})
    assert r.json()["rule"]["title"] == "改过的标题"
    bad = client.post("/api/rules", json={**RULE, "id": "X1", "airline": "ZZ"})
    assert bad.status_code == 400 and "airline" in bad.json()["error"]["fields"]

    r = client.post("/api/rules/search-debug", json={"query": "折扣经济舱改签手续费",
                                                     "filters": {"airline": "MU"}, "user_id": "u_zhang"})
    body = r.json()
    assert r.status_code == 200
    assert [s["stage"] for s in body["funnel"]] == ["dense", "sparse", "rrf", "filter", "rerank", "mmr", "inject"]
    assert body["rules"] and all(x["airline"] in ("MU", "*") for x in body["rules"])
    assert "airlines" in body["boost"]

    assert client.delete("/api/rules/TEST-PET").status_code == 200
    assert client.delete("/api/rules/TEST-PET").status_code == 404


# ---------------- 对话 SSE ----------------
def test_chat_sse_event_order(client, fake_graph):
    o = first_order(client)
    r = client.post("/api/chat", json={"session_id": "s1", "user_id": "u_zhang", "order_id": o["order_id"],
                                       "message": "帮我改到大后天"})
    assert r.status_code == 200 and r.headers["content-type"].startswith("text/event-stream")
    events = parse_sse(r.text)
    assert [e for e, _ in events] == ["trace", "trace", "interrupt", "final"]
    assert events[0][1]["node"] == "user_input"
    assert events[2][1]["kind"] == "confirm"
    assert events[3][1]["outcome"] == "ANSWER"
    assert fake_graph["run_turn"]["order_id"] == o["order_id"]


def test_chat_validation(client, fake_graph):
    r = client.post("/api/chat", json={"session_id": "s 1", "user_id": "u_zhang", "message": "hi"})
    assert r.status_code == 422 and "session_id" in r.json()["error"]["fields"]
    r = client.post("/api/chat", json={"session_id": "s1", "user_id": "u_zhang", "message": "x" * 1001})
    assert r.status_code == 422 and "message" in r.json()["error"]["fields"]
    r = client.post("/api/chat", json={"session_id": "s1", "user_id": "u_zhang", "message": "hi", "extra": 1})
    assert r.status_code == 422
    r = client.post("/api/chat", json={"session_id": "s1", "user_id": "u_nobody", "message": "hi"})
    assert r.status_code == 404


def test_chat_llm_not_configured(client, monkeypatch, fake_graph):
    from air_agent import llm
    import air_agent.graph as g

    def boom(*a, **k):
        raise llm.LLMNotConfigured("DeepSeek 模式需要先在配置中心填写 API Key")
    monkeypatch.setattr(g, "run_turn", boom)
    r = client.post("/api/chat", json={"session_id": "s2", "user_id": "u_zhang", "message": "hi", "mode": "deepseek"})
    events = parse_sse(r.text)
    assert events == [("error", {"code": "LLM_NOT_CONFIGURED", "message": "DeepSeek 模式需要先在配置中心填写 API Key"})]


def test_resume_and_session(client, fake_graph):
    r = client.post("/api/resume", json={"session_id": "s1", "payload": {"confirm": True, "index": 0,
                                                                         "confirm_id": "c-1"}})
    events = parse_sse(r.text)
    assert [e for e, _ in events] == ["trace", "final"]
    assert events[-1][1]["outcome"] == "TICKETED"
    assert fake_graph["resume"]["payload"]["confirm_id"] == "c-1"
    s = client.get("/api/sessions/s1").json()
    assert s["pending_interrupt"] == {"kind": "confirm"} and "order_stack" in s["memory"]


# ---------------- 评测 ----------------
CASE = {"id": "c01", "title": "改签差价", "tags": ["change"], "user_id": "u_zhang", "airline": "MU",
        "fare_product": "Y_DISCOUNT", "order": "#1", "question": "帮我改到大后天", "history": [],
        "clarify_answers": [], "expect": {"intents": ["CHANGE"], "tools": ["tool_rule_retrieve", "tool_recalc_fee"],
                                          "fees": {"CHANGE": 90}, "rule_ids": ["MU-Y_DISCOUNT-CHG"],
                                          "outcome": "ANSWER"}, "forbid": []}


def test_eval_cases_get_put_validate(client, monkeypatch):
    assert client.get("/api/eval/cases").json()["cases"] == []
    assert client.put("/api/eval/cases", json={"cases": [CASE]}).json()["count"] == 1
    assert client.get("/api/eval/cases").json()["cases"][0]["id"] == "c01"
    client.put("/api/eval/cases", json={"cases": [CASE, {**CASE, "id": "c02"}]})
    assert web.EVAL_CASES.with_suffix(".json.bak").exists()

    r = client.put("/api/eval/cases", json={"cases": [CASE, CASE]})
    assert r.status_code == 400 and "c01" in r.json()["error"]["fields"]["cases"]

    seen = {}

    def fake_cli(args, timeout=120):
        seen["args"] = args
        cases = json.loads(open(args[args.index("--cases") + 1], encoding="utf-8").read())
        errs = {c["id"]: [{"field": "expect.fees.CHANGE", "message": "期望金额与引擎不符，引擎值为 80"}]
                for c in cases if c["expect"]["fees"].get("CHANGE") == 999}
        return 0, json.dumps({"type": "validated", "errors": errs}) + "\n", ""
    monkeypatch.setattr(web, "_run_cli", fake_cli)
    r = client.post("/api/eval/cases/validate", json={"cases": [CASE]}).json()
    assert r == {"ok": True, "errors": {}, "source": "engine"} and seen["args"][0] == "validate"
    bad = {**CASE, "expect": {**CASE["expect"], "fees": {"CHANGE": 999}}}
    r = client.post("/api/eval/cases/validate", json={"cases": [bad]}).json()
    assert r["ok"] is False and r["errors"]["c01"][0]["field"] == "expect.fees.CHANGE"
    r = client.post("/api/eval/cases/validate", json={"cases": [{"id": "c9"}]}).json()
    assert r["source"] == "shape" and r["errors"]["c9"]


def test_eval_runs_list_and_get(client, tmp_path, monkeypatch):
    d = tmp_path / "eval_runs"
    d.mkdir(parents=True)
    monkeypatch.setattr(web, "RUNS_DIR", d)
    (d / "r1.json").write_text(json.dumps({"id": "r1", "created_at": "2026-09-26T10:00", "mode": "offline",
                                           "summary": {"n": 1}, "results": []}), "utf-8")
    runs = client.get("/api/eval/runs").json()
    assert runs["runs"][0]["id"] == "r1" and runs["active"] is None
    assert client.get("/api/eval/runs/r1").json()["summary"]["n"] == 1
    assert client.get("/api/eval/runs/nope").status_code == 404


def test_status_and_static(client):
    s = client.get("/api/status").json()
    assert s["mode"] == "offline" and s["retrieval"]["backend"] == "fallback"
    r = client.get("/")
    assert r.status_code == 200 and "退改签助手" in r.text
    assert client.get("/api/nope").json()["error"]["code"] == "NOT_FOUND"
