from datetime import datetime
from air_agent import settings as S


def test_defaults_and_update(tmp_path, monkeypatch):
    monkeypatch.setattr(S, "CONFIG_PATH", tmp_path / "config.json")
    S.reset()
    cfg = S.get()
    assert cfg["mode"] == "offline"
    assert cfg["verify"]["threshold"] == 0.85
    assert cfg["retrieval"]["k_recall"] == 60
    S.update({"verify": {"threshold": 0.9}})
    assert S.get()["verify"]["threshold"] == 0.9
    assert S.get()["verify"]["max_retries"] == 1   # 深合并，没改的保留
    assert (tmp_path / "config.json").exists()


def test_update_rejects_unknown_key(tmp_path, monkeypatch):
    monkeypatch.setattr(S, "CONFIG_PATH", tmp_path / "config.json")
    S.reset()
    import pytest
    with pytest.raises(KeyError):
        S.update({"nope": 1})


def test_now_override():
    S.set_now(datetime(2026, 9, 26, 10, 0))
    assert S.now() == datetime(2026, 9, 26, 10, 0)
    S.set_now(None)
    assert S.now().year >= 2026


def test_api_key_masked(tmp_path, monkeypatch):
    monkeypatch.setattr(S, "CONFIG_PATH", tmp_path / "config.json")
    S.reset()
    S.update({"llm": {"api_key": "sk-1234567890abcd"}})
    assert S.public()["llm"]["api_key"] == "sk-1****abcd"
    assert S.get()["llm"]["api_key"] == "sk-1234567890abcd"
