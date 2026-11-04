"""运行配置：默认值 + config.json 覆盖 + 运行时修改（配置中心调用 update）。"""
import copy
import json
import os
import threading
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT / "config.json"
VAR_DIR = Path(os.environ.get("AIR_VAR_DIR", ROOT / "var"))

DEFAULTS = {
    "mode": "offline",                      # offline | deepseek
    "llm": {"base_url": "https://api.deepseek.com", "api_key": "", "model": "deepseek-chat",
            "timeout": 30},
    "query": {"clarify_threshold": 0.6, "step_back_min_hits": 2},
    "planner": {"max_rounds": 5, "max_tool_retries": 1},
    "retrieval": {"backend": os.environ.get("AIR_RETRIEVAL", "hybrid"),   # hybrid | fallback
                  "k_recall": 60, "k_fuse": 40, "k_rerank": 10, "k_mmr": 5, "k_inject": 3,
                  "rrf_k": 60, "rerank_min": 0.6, "mmr_lambda": 0.7, "profile_boost": 0.1,
                  "dense_min": {"hybrid": 0.5, "fallback": 0.25}},
    "memory": {"recent_turns": 4, "order_stack": 5, "token_budget": 3000,
               "budget_share": {"system": 0.15, "order": 0.15, "rules": 0.4,
                                "history": 0.2, "profile": 0.1}},
    "verify": {"threshold": 0.85, "max_retries": 1},
}

_lock = threading.Lock()
_cfg = None
_now = datetime.fromisoformat(os.environ["AIR_NOW"]) if os.environ.get("AIR_NOW") else None


def _merge(base, patch, path=""):
    for k, v in patch.items():
        if k not in base:
            raise KeyError(f"未知配置项：{path}{k}")
        if isinstance(base[k], dict) and isinstance(v, dict) and k != "budget_share":
            _merge(base[k], v, f"{path}{k}.")
        else:
            base[k] = v


def _load():
    cfg = copy.deepcopy(DEFAULTS)
    if CONFIG_PATH.exists():
        _merge(cfg, json.loads(CONFIG_PATH.read_text("utf-8")))
    return cfg


def get():
    global _cfg
    with _lock:
        if _cfg is None:
            _cfg = _load()
        return _cfg


def update(patch: dict):
    global _cfg
    with _lock:
        cfg = copy.deepcopy(_cfg if _cfg is not None else _load())
        _merge(cfg, patch)
        _cfg = cfg
        CONFIG_PATH.write_text(json.dumps(cfg, ensure_ascii=False, indent=2), "utf-8")
        return cfg


def reset():
    global _cfg
    with _lock:
        _cfg = copy.deepcopy(DEFAULTS)
        if CONFIG_PATH.exists():
            CONFIG_PATH.unlink()


def public():
    """给前端看的配置：API Key 打码。"""
    cfg = copy.deepcopy(get())
    key = cfg["llm"]["api_key"]
    cfg["llm"]["api_key"] = f"{key[:4]}****{key[-4:]}" if len(key) > 8 else ("****" if key else "")
    return cfg


def set_now(value):
    global _now
    _now = value


def now() -> datetime:
    return _now or datetime.now().replace(second=0, microsecond=0)
