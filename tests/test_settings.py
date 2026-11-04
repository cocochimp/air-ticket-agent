import json
from datetime import datetime

import pytest

from air_agent import settings as S


def test_defaults_and_update():
    S.reset()
    cfg = S.get()
    assert cfg["mode"] == "offline"
    assert cfg["verify"]["threshold"] == 0.85
    assert cfg["retrieval"]["k_recall"] == 60
    S.update({"verify": {"threshold": 0.9}})
    assert S.get()["verify"]["threshold"] == 0.9
    assert S.get()["verify"]["max_retries"] == 1   # 深合并，没改的保留
    assert S.CONFIG_PATH.exists()


def test_update_persists_only_overrides():
    S.reset()
    S.update({"verify": {"threshold": 0.9}})
    assert json.loads(S.CONFIG_PATH.read_text("utf-8")) == {"verify": {"threshold": 0.9}}
    S.update({"planner": {"max_rounds": 7}})
    assert json.loads(S.CONFIG_PATH.read_text("utf-8")) == {
        "verify": {"threshold": 0.9}, "planner": {"max_rounds": 7}}
    assert S.get()["planner"]["max_tool_retries"] == 1


def test_update_rejects_unknown_key():
    S.reset()
    with pytest.raises(KeyError):
        S.update({"nope": 1})


def test_update_rejects_wrong_types():
    S.reset()
    with pytest.raises(TypeError):
        S.update({"verify": "oops"})
    with pytest.raises(TypeError):
        S.update({"verify": {"threshold": "high"}})
    with pytest.raises(TypeError):
        S.update({"verify": {"max_retries": True}})
    with pytest.raises(TypeError):
        S.update({"mode": 1})
    assert S.get()["verify"]["threshold"] == 0.85     # 失败的更新不留痕迹
    assert not S.CONFIG_PATH.exists()


def test_update_accepts_int_for_float_default():
    S.reset()
    S.update({"verify": {"threshold": 1}})
    assert S.get()["verify"]["threshold"] == 1


def test_now_override():
    S.set_now(datetime(2026, 9, 26, 10, 0))
    assert S.now() == datetime(2026, 9, 26, 10, 0)
    S.set_now(None)
    n = S.now()
    assert isinstance(n, datetime) and n.second == 0


def test_api_key_masked():
    S.reset()
    S.update({"llm": {"api_key": "sk-1234567890abcd"}})
    assert S.public()["llm"]["api_key"] == "sk-1****abcd"
    assert S.get()["llm"]["api_key"] == "sk-1234567890abcd"
