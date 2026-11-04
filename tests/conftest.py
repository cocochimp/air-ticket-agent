import os
os.environ.setdefault("AIR_RETRIEVAL", "fallback")

from datetime import datetime
import pytest
from air_agent import settings as S

FIXED_NOW = datetime(2026, 9, 26, 10, 0)


@pytest.fixture(autouse=True)
def _isolated(tmp_path, monkeypatch):
    monkeypatch.setattr(S, "CONFIG_PATH", tmp_path / "config.json")
    monkeypatch.setattr(S, "VAR_DIR", tmp_path / "var")
    S.reset()
    S.update({"retrieval": {"backend": "fallback"}})
    S.set_now(FIXED_NOW)
    yield
    S.set_now(None)
