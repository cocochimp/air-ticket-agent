"""运行配置：默认值 + config.json 覆盖 + 运行时修改（配置中心调用 update）。

config.json 只保存相对默认值的覆盖项（_overrides），生效配置 = DEFAULTS 与覆盖项合并。
"""
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
                  "dense_min": {"hybrid": 0.6, "fallback": 0.25},
                  "dense_strict": {"hybrid": 0.7, "fallback": 0.45}},
    "memory": {"recent_turns": 4, "order_stack": 5, "token_budget": 3000,
               "budget_share": {"system": 0.15, "order": 0.15, "rules": 0.4,
                                "history": 0.2, "profile": 0.1}},
    "verify": {"threshold": 0.85, "max_retries": 1},
}

_lock = threading.Lock()
_cfg = None          # 生效配置
_overrides = {}      # 相对 DEFAULTS 的覆盖项，只有它会写入 config.json
_now = datetime.fromisoformat(os.environ["AIR_NOW"]) if os.environ.get("AIR_NOW") else None


def _check_type(default, value, path):
    if isinstance(default, dict):
        ok = isinstance(value, dict)
    elif isinstance(default, bool):
        ok = isinstance(value, bool)
    elif isinstance(default, (int, float)):
        ok = isinstance(value, (int, float)) and not isinstance(value, bool)
    elif isinstance(default, str):
        ok = isinstance(value, str)
    else:
        ok = True
    if not ok:
        raise TypeError(f"配置项 {path} 类型错误：期望 {type(default).__name__}，实际 {type(value).__name__}")


def _merge(base, patch, path=""):
    for k, v in patch.items():
        if k not in base:
            raise KeyError(f"未知配置项：{path}{k}")
        _check_type(base[k], v, f"{path}{k}")
        if isinstance(base[k], dict) and k != "budget_share":
            _merge(base[k], v, f"{path}{k}.")
        else:
            base[k] = copy.deepcopy(v)


def _deep_set(target, patch):
    """把已校验的 patch 并入覆盖项；budget_share 整体替换。"""
    for k, v in patch.items():
        if isinstance(v, dict) and k != "budget_share":
            _deep_set(target.setdefault(k, {}), v)
        else:
            target[k] = copy.deepcopy(v)


MIN_TOKEN_BUDGET = 200      # 系统提示 + 订单信息必须放得下，且不能被截断


def _validate(cfg):
    if cfg["memory"]["token_budget"] < MIN_TOKEN_BUDGET:
        raise ValueError(f"memory.token_budget 不能小于 {MIN_TOKEN_BUDGET}")


def _load():
    cfg = copy.deepcopy(DEFAULTS)
    overrides = json.loads(CONFIG_PATH.read_text("utf-8")) if CONFIG_PATH.exists() else {}
    _merge(cfg, overrides)
    _validate(cfg)
    return cfg, overrides


def _ensure():
    global _cfg, _overrides
    if _cfg is None:
        _cfg, _overrides = _load()


def get():
    with _lock:
        _ensure()
        return _cfg


def update(patch: dict):
    global _cfg, _overrides
    with _lock:
        _ensure()
        cfg = copy.deepcopy(_cfg)
        _merge(cfg, patch)                      # 先校验，失败则不改任何状态
        _validate(cfg)
        overrides = copy.deepcopy(_overrides)
        _deep_set(overrides, patch)
        CONFIG_PATH.write_text(json.dumps(overrides, ensure_ascii=False, indent=2), "utf-8")
        _cfg, _overrides = cfg, overrides
        return cfg


def reset():
    """仅供测试使用：会删除 config.json。"""
    global _cfg, _overrides
    with _lock:
        _cfg = copy.deepcopy(DEFAULTS)
        _overrides = {}
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
