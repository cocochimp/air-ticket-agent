# 阶段一：Agent 核心 实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal：** 实现机票退改签 Agent 的后端核心（运价引擎、5 个工具、Query 理解、混合检索、记忆、质量校验、LangGraph 状态图、评测），规则模式离线可跑通，DeepSeek 模式可接入。

**Architecture：** `air_agent/` 包内每个文件对应设计文档的一节，模块之间通过 Pydantic 模型和 `AgentState` 传数据。LangGraph `StateGraph` 负责编排，SqliteSaver 保存检查点。所有金额只来自 `engine.py`。

**Tech Stack：** Python 3.13、uv、LangGraph 1.2、Pydantic 2、fastembed（BGE-small-zh）、qdrant-client（本地模式）、rank-bm25、jieba、flashrank、pytest。

**设计文档：** `docs/superpowers/specs/2026-11-04-air-ticket-agent-design.md`

**后续计划：** 阶段二（FastAPI + 对话工作台）、阶段三（评测中心 / 规则库 / 订单与用户 / 配置中心）各写一份计划，在本阶段完成后编写。

**环境说明：**
- PyPI 走阿里云镜像（已写进 `pyproject.toml`）。
- HuggingFace 模型走 `HF_ENDPOINT=https://hf-mirror.com`。
- 模型缓存放在 `models/`，不提交到 git。
- 单元测试默认使用 `AIR_RETRIEVAL=fallback`（TF-IDF + BM25），不下载模型；`tests/test_retrieval_models.py` 单独测真实模型，标记为 `slow`。

---

## 文件结构

| 文件 | 职责 |
|---|---|
| `pyproject.toml` | 依赖、pytest 配置、镜像源 |
| `air_agent/settings.py` | 默认配置 + `config.json` 覆盖 + 运行时修改；`now()` 可注入 |
| `air_agent/seed.py` | 按「今天」生成虚构的航司、航班、订单、用户、规则（确定性生成） |
| `air_agent/data.py` | SQLite 连接、建表、写入种子数据、各类查询函数 |
| `air_agent/engine.py` | 运价引擎：改签费、差价、退票费、可行性判断 |
| `air_agent/state.py` | `AgentState`、`ParsedQuery`、`SubQuery`、`SchemeOutput` 等模型 |
| `air_agent/tools.py` | 5 个工具的参数 Schema、`run_tool()`、错误反馈结构 |
| `air_agent/query.py` | 订单字段对齐、意图与多诉求拆解、指代消解、日期换算、Step-back、HyDE |
| `air_agent/retrieval.py` | 检索漏斗及降级 |
| `air_agent/memory.py` | 短期会话记忆、长期画像、Token 预算 |
| `air_agent/verify.py` | 金额比对、事实核查、引用合法性、置信度 |
| `air_agent/scheme.py` | 方案生成（模板 / LLM），金额回填 |
| `air_agent/llm.py` | DeepSeek 客户端、`FakeLLM` |
| `air_agent/planner.py` | ReAct 决策：规则计划表 / LLM Function Calling |
| `air_agent/graph.py` | StateGraph 节点、条件边、interrupt、`run_turn()` / `resume()` |
| `air_agent/evaluation.py` | 用例校验、批量评测、指标、CLI |
| `eval/gold_cases.json` | 金标准用例 |
| `tests/` | 每个模块一个测试文件 |

---

### Task 1：项目脚手架与配置

**Files:**
- Create: `pyproject.toml`、`.gitignore`、`air_agent/__init__.py`、`air_agent/settings.py`、`tests/conftest.py`
- Test: `tests/test_settings.py`

- [ ] **Step 1：写 `pyproject.toml` 和 `.gitignore`**

```toml
[project]
name = "air-ticket-agent"
version = "0.1.0"
description = "机票退改签 Agent MVP（LangGraph）"
requires-python = ">=3.13"
dependencies = [
  "langgraph>=1.0,<2",
  "langgraph-checkpoint-sqlite>=3,<4",
  "fastapi>=0.115,<1",
  "uvicorn>=0.30,<1",
  "pydantic>=2.9,<3",
  "httpx>=0.28,<1",
  "fastembed>=0.7,<1",
  "qdrant-client>=1.14,<2",
  "rank-bm25>=0.2.2",
  "jieba>=0.42",
  "flashrank>=0.2,<1",
]

[dependency-groups]
dev = ["pytest>=8"]

[tool.uv]
index-url = "https://mirrors.aliyun.com/pypi/simple"

[tool.pytest.ini_options]
testpaths = ["tests"]
markers = ["slow: 需要下载/加载真实检索模型"]
addopts = "-m 'not slow'"
```

```gitignore
.venv/
__pycache__/
*.pyc
models/
var/
config.json
.pytest_cache/
```

- [ ] **Step 2：写失败测试 `tests/test_settings.py`**

```python
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
```

- [ ] **Step 3：运行测试，确认失败**

Run: `uv sync && uv run pytest tests/test_settings.py -v`
Expected: FAIL，`ModuleNotFoundError: No module named 'air_agent'`

- [ ] **Step 4：实现 `air_agent/__init__.py`（空文件）和 `air_agent/settings.py`**

```python
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
```

- [ ] **Step 5：写 `tests/conftest.py`（后续测试共用）**

```python
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
```

- [ ] **Step 6：运行测试，确认通过**

Run: `uv run pytest tests/test_settings.py -v`
Expected: 4 passed

- [ ] **Step 7：提交**

```bash
git add pyproject.toml uv.lock .gitignore air_agent tests
git commit -m "feat: scaffold project and runtime settings"
```

### Task 2：虚构种子数据与 SQLite 数据层

**Files:**
- Create: `air_agent/seed.py`、`air_agent/data.py`
- Test: `tests/test_data.py`

种子数据以「今天」为基准生成，保证演示时总有未来航班。同一天生成的数据完全一致。

- [ ] **Step 1：写失败测试 `tests/test_data.py`**

```python
from air_agent import data, seed


def test_init_seeds_everything():
    data.init()
    orders = data.list_orders()
    assert len(orders) == 12
    assert {o["user_id"] for o in orders} == {"u_zhang", "u_li", "u_wang"}
    assert all(o["order_id"].startswith("ORD") and len(o["order_id"]) == 14 for o in orders)
    rules = data.list_rules()
    assert len(rules) == 4 * 6 * 2 + 4
    assert data.get_rule("MU-Y_DISCOUNT-CHG")["scenario"] == "VOLUNTARY_CHANGE"
    assert len(data.list_profiles()) == 3


def test_init_is_idempotent():
    data.init()
    data.init()
    assert len(data.list_orders()) == 12


def test_flight_status_update():
    data.init()
    o = data.list_orders("u_zhang")[0]
    data.set_flight_status(o["flight_no"], o["date"], "delayed", 150)
    f = data.get_flight(o["flight_no"], o["date"])
    assert f["status"] == "delayed" and f["delay_min"] == 150


def test_seed_scenarios_present():
    data.init()
    statuses = {data.get_flight(o["flight_no"], o["date"])["status"] for o in data.list_orders()}
    assert {"scheduled", "cancelled", "delayed"} <= statuses
    products = {o["fare_product"] for o in data.list_orders()}
    assert products == set(seed.PRODUCTS)


def test_rule_text_matches_fee_table():
    data.init()
    rule = data.get_rule("CA-Y-RFD")
    pct = data.get_fee_rule("CA", "Y")["refund"][0]
    assert f"{pct}%" in rule["content"]
```

- [ ] **Step 2：运行测试，确认失败**

Run: `uv run pytest tests/test_data.py -v`
Expected: FAIL，`ImportError: cannot import name 'data'`

- [ ] **Step 3：实现 `air_agent/seed.py`**

```python
"""确定性生成虚构演示数据：航司、航班、订单、用户画像、费率表与退改规则。"""
from datetime import datetime, timedelta

AIRLINES = {"MU": "东方云航", "CA": "国际星航", "CZ": "南方海航", "HU": "海岛航空"}
# fare_product -> (舱位大类, 中文名)
PRODUCTS = {"F": ("F", "头等舱全价"), "C": ("C", "公务舱全价"), "Y": ("Y", "经济舱全价"),
            "Y_DISCOUNT": ("Y", "经济舱折扣"), "SPECIAL": ("Y", "经济舱特价"), "GROUP": ("Y", "团队票")}
PRODUCT_FACTOR = {"F": 3.0, "C": 2.0, "Y": 1.0, "Y_DISCOUNT": 0.6, "SPECIAL": 0.35, "GROUP": 0.5}
WINDOWS = ["d7", "h48", "h4", "lt4", "after"]
WINDOW_LABEL = {"d7": "起飞前7天(含)以上", "h48": "起飞前48小时(含)至7天",
                "h4": "起飞前4小时(含)至48小时", "lt4": "起飞前4小时以内", "after": "起飞后"}
# 费率（票面价百分比），按 WINDOWS 顺序；None 表示不允许
BASE_RATES = {
    "F": {"change": [0, 0, 5, 10, 20], "refund": [0, 5, 10, 15, 25]},
    "C": {"change": [0, 5, 10, 15, 25], "refund": [5, 10, 15, 20, 30]},
    "Y": {"change": [5, 10, 15, 20, 30], "refund": [10, 15, 20, 25, 40]},
    "Y_DISCOUNT": {"change": [10, 20, 30, 40, 50], "refund": [20, 30, 40, 50, 70]},
    "SPECIAL": {"change": None, "refund": [50, 60, 70, 80, 100]},
    "GROUP": {"change": None, "refund": None},
}
AIRLINE_OFFSET = {"MU": 0, "CA": 5, "CZ": 0, "HU": 5}
ROUTES = [("HGH", "PEK", "杭州", "北京", 1500), ("PEK", "HGH", "北京", "杭州", 1500),
          ("SHA", "CAN", "上海", "广州", 1300), ("PEK", "CTU", "北京", "成都", 1600),
          ("CAN", "HAK", "广州", "海口", 900)]
DAY_MULT = [1.0, 1.08, 0.95, 1.15, 1.0, 1.2, 0.9]
TAX = 50


def fee_table():
    out = {}
    for al, off in AIRLINE_OFFSET.items():
        for prod, rates in BASE_RATES.items():
            out[(al, prod)] = {k: (None if v is None else [min(100, p + off) if p > 0 else 0 for p in v])
                               for k, v in rates.items()}
    return out


def flight_no(ai: int, ri: int, al: str) -> str:
    return f"{al}{5101 + ri * 10 + ai}"


def fare_for(base: int, prod: str, day, ai: int) -> int:
    mult = DAY_MULT[(day.toordinal() + ai) % 7]
    return int(round(base * PRODUCT_FACTOR[prod] * mult / 10) * 10)


def flights(today):
    rows = []
    for ai, al in enumerate(AIRLINES):
        for ri, (o, d, oc, dc, base) in enumerate(ROUTES):
            dep_h = 7 + (ri * 3 + ai * 2) % 14
            for off in range(-1, 15):
                day = today + timedelta(days=off)
                rows.append({"flight_no": flight_no(ai, ri, al), "date": day.isoformat(), "airline": al,
                             "origin": o, "dest": d, "origin_city": oc, "dest_city": dc,
                             "dep_time": f"{dep_h:02d}:{ai * 15:02d}", "duration_min": 130,
                             "status": "scheduled", "delay_min": 0,
                             "fares": {p: fare_for(base, p, day, ai) for p in PRODUCTS}})
    return rows


# (user, airline, route_idx, product, day_offset, passenger, flight_status, delay)
ORDER_SPECS = [
    ("u_zhang", "MU", 0, "Y_DISCOUNT", 2, "张三", "scheduled", 0),
    ("u_zhang", "MU", 1, "Y_DISCOUNT", 5, "张三", "scheduled", 0),
    ("u_zhang", "CA", 2, "C", 10, "张三", "scheduled", 0),
    ("u_li", "CZ", 4, "SPECIAL", 1, "李四", "scheduled", 0),
    ("u_li", "HU", 3, "Y", 0, "李四", "scheduled", 0),        # 今天，起飞时间改为 now+3h
    ("u_li", "CA", 3, "F", 8, "李四", "scheduled", 0),
    ("u_wang", "MU", 2, "Y", 1, "王五", "cancelled", 0),
    ("u_wang", "CZ", 0, "Y_DISCOUNT", 3, "王五", "delayed", 150),
    ("u_wang", "HU", 4, "GROUP", 4, "王五", "scheduled", 0),
    ("u_zhang", "CZ", 2, "Y", -1, "张三", "scheduled", 0),    # 昨天，已误机
    ("u_li", "MU", 0, "Y", 6, "李四", "scheduled", 0),
    ("u_wang", "CA", 0, "SPECIAL", 2, "王五", "scheduled", 0),
]

PROFILES = [
    {"user_id": "u_zhang", "name": "张三", "tier": "金卡",
     "data": {"routes": {"HGH-PEK": 8, "PEK-HGH": 6}, "airlines": {"MU": 10}, "products": {"Y_DISCOUNT": 9},
              "prefer_action": {"CHANGE": 5, "REFUND": 1}, "price_sensitivity": "高"}},
    {"user_id": "u_li", "name": "李四", "tier": "银卡",
     "data": {"routes": {"PEK-CTU": 4}, "airlines": {"HU": 3, "CA": 2}, "products": {"Y": 3, "F": 2},
              "prefer_action": {"REFUND": 2}, "price_sensitivity": "中"}},
    {"user_id": "u_wang", "name": "王五", "tier": "普卡",
     "data": {"routes": {}, "airlines": {}, "products": {}, "prefer_action": {}, "price_sensitivity": "低"}},
]


def orders(today, now: datetime, flight_rows):
    by_key = {(f["flight_no"], f["date"]): f for f in flight_rows}
    out = []
    for i, (uid, al, ri, prod, off, pax, status, delay) in enumerate(ORDER_SPECS, start=1):
        ai = list(AIRLINES).index(al)
        f = by_key[(flight_no(ai, ri, al), (today + timedelta(days=off)).isoformat())]
        if off == 0:
            dep = now + timedelta(hours=3)
            f["dep_time"] = f"{dep.hour:02d}:{dep.minute:02d}" if dep.date() == today else "23:50"
        f["status"], f["delay_min"] = status, delay
        out.append({"order_id": f"ORD{today:%Y%m%d}{i:03d}", "user_id": uid, "passenger": pax,
                    "flight_no": f["flight_no"], "date": f["date"], "airline": al,
                    "cabin": PRODUCTS[prod][0], "fare_product": prod, "fare": f["fares"][prod],
                    "tax": TAX, "status": "ISSUED", "ticket_no": f"781-{2400000000 + i}"})
    return out


def _pct_text(p, verb):
    return f"免收{verb}费" if p == 0 else f"收取票面价{p}%的{verb}费"


def rules():
    out = []
    for (al, prod), table in fee_table().items():
        name, (cabin, plabel) = AIRLINES[al], PRODUCTS[prod]
        if table["change"] is None:
            chg = "该票价产品不得自愿改签，如需变更行程请办理退票后重新购票。"
        else:
            chg = "；".join(f"{WINDOW_LABEL[w]}申请改签，{_pct_text(p, '改期')}"
                           for w, p in zip(WINDOWS, table["change"]))
            chg += "。改签至票面价更高的航班需补齐差价，改签至票面价更低的航班不退差额。"
        if table["refund"] is None:
            rfd = "该票价产品不得自愿退票。"
        else:
            rfd = "；".join(f"{WINDOW_LABEL[w]}申请退票，{_pct_text(p, '退票')}"
                           for w, p in zip(WINDOWS, table["refund"]))
            rfd += f"。机场建设费{TAX}元在退票时全额退还。"
        for suffix, scen, kind, text in (("CHG", "VOLUNTARY_CHANGE", "自愿改签", chg),
                                         ("RFD", "VOLUNTARY_REFUND", "自愿退票", rfd)):
            out.append({"id": f"{al}-{prod}-{suffix}", "airline": al, "cabin": cabin, "fare_product": prod,
                        "scenario": scen, "title": f"{name}{plabel}（{prod}）{kind}规则", "content": text})
    out += [
        {"id": "GEN-CANCEL", "airline": "*", "cabin": "*", "fare_product": "*", "scenario": "CANCEL",
         "title": "航班取消（非旅客原因）退改规则",
         "content": "因航空公司原因航班取消，旅客可免费改签至同航线其他航班，或办理全额退票，免收改期费和退票费，改签不收取差价。"},
        {"id": "GEN-DELAY", "airline": "*", "cabin": "*", "fare_product": "*", "scenario": "DELAY",
         "title": "航班延误退改规则",
         "content": "航班延误2小时(含)以上，旅客可免费改签或全额退票，免收改期费和退票费，改签不收取差价；延误不足2小时按自愿退改规则执行。"},
        {"id": "GEN-NOSHOW", "airline": "*", "cabin": "*", "fare_product": "*", "scenario": "NO_SHOW",
         "title": "误机处理规则",
         "content": "旅客未在航班起飞前办理退改的视为误机，按各票价产品「起飞后」档位收取改期费或退票费。"},
        {"id": "GEN-FAREDIFF", "airline": "*", "cabin": "*", "fare_product": "*", "scenario": "FARE_DIFF",
         "title": "改签差价计算规则",
         "content": "改签差价按改签后航班同一票价产品的票面价减去原票面价计算，差价为正需补齐，差价为负不退还；改期费与差价分别计收。"},
    ]
    return out
```

- [ ] **Step 4：实现 `air_agent/data.py`**

```python
"""SQLite 数据层：建表、写入种子数据、查询与修改。"""
import json
import sqlite3
import threading
from contextlib import contextmanager
from . import seed
from . import settings as S

_init_lock = threading.Lock()

SCHEMA = """
CREATE TABLE IF NOT EXISTS flights(flight_no TEXT, date TEXT, airline TEXT, origin TEXT, dest TEXT,
  origin_city TEXT, dest_city TEXT, dep_time TEXT, duration_min INT, status TEXT, delay_min INT,
  fares TEXT, PRIMARY KEY(flight_no, date));
CREATE TABLE IF NOT EXISTS orders(order_id TEXT PRIMARY KEY, user_id TEXT, passenger TEXT, flight_no TEXT,
  date TEXT, airline TEXT, cabin TEXT, fare_product TEXT, fare INT, tax INT, status TEXT, ticket_no TEXT);
CREATE TABLE IF NOT EXISTS profiles(user_id TEXT PRIMARY KEY, name TEXT, tier TEXT, data TEXT);
CREATE TABLE IF NOT EXISTS fee_rules(airline TEXT, fare_product TEXT, data TEXT, PRIMARY KEY(airline, fare_product));
CREATE TABLE IF NOT EXISTS rules(id TEXT PRIMARY KEY, airline TEXT, cabin TEXT, fare_product TEXT,
  scenario TEXT, title TEXT, content TEXT, updated_at TEXT);
CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY, value TEXT);
"""
JSON_COLS = {"fares", "data"}


def db_path():
    S.VAR_DIR.mkdir(parents=True, exist_ok=True)
    return S.VAR_DIR / "air.db"


@contextmanager
def connect():
    conn = sqlite3.connect(db_path(), timeout=10)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def _row(r):
    if r is None:
        return None
    d = dict(r)
    for k in JSON_COLS & d.keys():
        d[k] = json.loads(d[k])
    return d


def init(force: bool = False):
    with _init_lock, connect() as c:
        c.executescript(SCHEMA)
        seeded = c.execute("SELECT value FROM meta WHERE key='seeded'").fetchone()
        if seeded and not force:
            return
        for t in ("flights", "orders", "profiles", "fee_rules", "rules"):
            c.execute(f"DELETE FROM {t}")
        now = S.now()
        today = now.date()
        fl = seed.flights(today)
        od = seed.orders(today, now, fl)
        c.executemany("INSERT INTO flights VALUES(:flight_no,:date,:airline,:origin,:dest,:origin_city,"
                      ":dest_city,:dep_time,:duration_min,:status,:delay_min,:fares)",
                      [{**f, "fares": json.dumps(f["fares"])} for f in fl])
        c.executemany("INSERT INTO orders VALUES(:order_id,:user_id,:passenger,:flight_no,:date,:airline,"
                      ":cabin,:fare_product,:fare,:tax,:status,:ticket_no)", od)
        c.executemany("INSERT INTO profiles VALUES(:user_id,:name,:tier,:data)",
                      [{**p, "data": json.dumps(p["data"], ensure_ascii=False)} for p in seed.PROFILES])
        c.executemany("INSERT INTO fee_rules VALUES(?,?,?)",
                      [(al, prod, json.dumps(t)) for (al, prod), t in seed.fee_table().items()])
        c.executemany("INSERT INTO rules VALUES(:id,:airline,:cabin,:fare_product,:scenario,:title,:content,:ts)",
                      [{**r, "ts": now.isoformat()} for r in seed.rules()])
        c.execute("INSERT OR REPLACE INTO meta VALUES('seeded', ?)", (now.isoformat(),))


def _all(sql, args=()):
    with connect() as c:
        return [_row(r) for r in c.execute(sql, args).fetchall()]


def _one(sql, args=()):
    with connect() as c:
        return _row(c.execute(sql, args).fetchone())


def get_order(order_id):
    return _one("SELECT * FROM orders WHERE order_id=?", (order_id,))


def list_orders(user_id=None):
    if user_id:
        return _all("SELECT * FROM orders WHERE user_id=? ORDER BY date", (user_id,))
    return _all("SELECT * FROM orders ORDER BY order_id")


def update_order(order_id, **fields):
    allowed = {"flight_no", "date", "fare", "status", "ticket_no"}
    assert set(fields) <= allowed, fields
    sets = ", ".join(f"{k}=?" for k in fields)
    with connect() as c:
        c.execute(f"UPDATE orders SET {sets} WHERE order_id=?", (*fields.values(), order_id))


def get_flight(flight_no, date):
    return _one("SELECT * FROM flights WHERE flight_no=? AND date=?", (flight_no, date))


def list_flights(origin=None, dest=None, date=None):
    sql, args = "SELECT * FROM flights WHERE 1=1", []
    for col, val in (("origin", origin), ("dest", dest), ("date", date)):
        if val:
            sql += f" AND {col}=?"
            args.append(val)
    return _all(sql + " ORDER BY date, dep_time", args)


def set_flight_status(flight_no, date, status, delay_min=0):
    assert status in {"scheduled", "delayed", "cancelled"}
    with connect() as c:
        c.execute("UPDATE flights SET status=?, delay_min=? WHERE flight_no=? AND date=?",
                  (status, delay_min if status == "delayed" else 0, flight_no, date))


def get_profile(user_id):
    return _one("SELECT * FROM profiles WHERE user_id=?", (user_id,))


def list_profiles():
    return _all("SELECT * FROM profiles ORDER BY user_id")


def save_profile(user_id, data):
    with connect() as c:
        c.execute("UPDATE profiles SET data=? WHERE user_id=?", (json.dumps(data, ensure_ascii=False), user_id))


def get_fee_rule(airline, fare_product):
    r = _one("SELECT data FROM fee_rules WHERE airline=? AND fare_product=?", (airline, fare_product))
    return r["data"] if r else None


def list_rules(airline=None, fare_product=None, scenario=None):
    sql, args = "SELECT * FROM rules WHERE 1=1", []
    for col, val in (("airline", airline), ("fare_product", fare_product), ("scenario", scenario)):
        if val:
            sql += f" AND {col}=?"
            args.append(val)
    return _all(sql + " ORDER BY id", args)


def get_rule(rule_id):
    return _one("SELECT * FROM rules WHERE id=?", (rule_id,))


def upsert_rule(rule: dict):
    with connect() as c:
        c.execute("INSERT OR REPLACE INTO rules VALUES(:id,:airline,:cabin,:fare_product,:scenario,:title,"
                  ":content,:ts)", {**rule, "ts": S.now().isoformat()})


def delete_rule(rule_id):
    with connect() as c:
        c.execute("DELETE FROM rules WHERE id=?", (rule_id,))
```

- [ ] **Step 5：运行测试，确认通过**

Run: `uv run pytest tests/test_data.py -v`
Expected: 5 passed

- [ ] **Step 6：提交**

```bash
git add air_agent/seed.py air_agent/data.py tests/test_data.py
git commit -m "feat: add fictional seed data and sqlite data layer"
```

### Task 3：运价引擎（唯一金额来源）

**Files:**
- Create: `air_agent/engine.py`
- Test: `tests/test_engine.py`

- [ ] **Step 1：写失败测试 `tests/test_engine.py`**

```python
from datetime import date, timedelta
import pytest
from air_agent import data, engine, seed
from air_agent import settings as S


def _order(pred):
    data.init()
    return next(o for o in data.list_orders() if pred(o))


def test_window_boundaries():
    now = S.now()
    assert engine.window(now + timedelta(days=7), now) == "d7"
    assert engine.window(now + timedelta(hours=48), now) == "h48"
    assert engine.window(now + timedelta(hours=4), now) == "h4"
    assert engine.window(now + timedelta(hours=3, minutes=59), now) == "lt4"
    assert engine.window(now - timedelta(minutes=1), now) == "after"


def test_change_fee_and_fare_diff():
    o = _order(lambda o: o["user_id"] == "u_zhang" and o["fare_product"] == "Y_DISCOUNT"
               and o["flight_no"].startswith("MU51") and o["flight_no"].endswith("01"))
    target = (date.fromisoformat(o["date"]) + timedelta(days=1)).isoformat()
    r = engine.calc(o["order_id"], "CHANGE", target_date=target)
    w = r["window"]
    pct = data.get_fee_rule("MU", "Y_DISCOUNT")["change"][seed.WINDOWS.index(w)]
    new_fare = data.get_flight(o["flight_no"], target)["fares"]["Y_DISCOUNT"]
    assert r["allowed"] and not r["involuntary"]
    assert r["change_fee"] == round(o["fare"] * pct / 100)
    assert r["fare_diff"] == max(0, new_fare - o["fare"])
    assert r["total"] == r["change_fee"] + r["fare_diff"]
    assert r["rule_id"] == "MU-Y_DISCOUNT-CHG"


def test_refund_amount_includes_tax():
    o = _order(lambda o: o["fare_product"] == "C")
    r = engine.calc(o["order_id"], "REFUND")
    assert r["refund_amount"] == o["fare"] - r["refund_fee"] + o["tax"]
    assert r["rule_id"] == "CA-C-RFD"


def test_cancelled_flight_is_free():
    o = _order(lambda o: data.get_flight(o["flight_no"], o["date"])["status"] == "cancelled")
    target = (date.fromisoformat(o["date"]) + timedelta(days=1)).isoformat()
    ch = engine.calc(o["order_id"], "CHANGE", target_date=target)
    rf = engine.calc(o["order_id"], "REFUND")
    assert ch["involuntary"] and ch["total"] == 0 and ch["rule_id"] == "GEN-CANCEL"
    assert rf["refund_fee"] == 0 and rf["refund_amount"] == o["fare"] + o["tax"]


def test_delay_over_2h_is_free():
    o = _order(lambda o: data.get_flight(o["flight_no"], o["date"])["status"] == "delayed")
    r = engine.calc(o["order_id"], "REFUND")
    assert r["involuntary"] and r["refund_fee"] == 0 and r["rule_id"] == "GEN-DELAY"


def test_special_cannot_change_group_cannot_refund():
    sp = _order(lambda o: o["fare_product"] == "SPECIAL")
    gp = _order(lambda o: o["fare_product"] == "GROUP")
    target = (date.fromisoformat(sp["date"]) + timedelta(days=1)).isoformat()
    assert engine.calc(sp["order_id"], "CHANGE", target_date=target)["allowed"] is False
    assert engine.calc(gp["order_id"], "REFUND")["allowed"] is False


def test_no_show_uses_after_window():
    o = _order(lambda o: o["date"] < S.now().date().isoformat())
    r = engine.calc(o["order_id"], "REFUND")
    assert r["window"] == "after"


def test_errors():
    data.init()
    with pytest.raises(engine.EngineError, match="订单不存在"):
        engine.calc("ORD00000000000", "REFUND")
    o = data.list_orders()[0]
    with pytest.raises(engine.EngineError, match="target_date"):
        engine.calc(o["order_id"], "CHANGE")
    with pytest.raises(engine.EngineError, match="没有可改签的航班"):
        engine.calc(o["order_id"], "CHANGE", target_date="2030-01-01")
```

- [ ] **Step 2：运行测试，确认失败**

Run: `uv run pytest tests/test_engine.py -v`
Expected: FAIL，`ImportError: cannot import name 'engine'`

- [ ] **Step 3：实现 `air_agent/engine.py`**

```python
"""运价引擎：改签费、差价、退票费。系统里所有金额都只能从这里来。"""
from datetime import datetime
from . import data, seed
from . import settings as S


class EngineError(Exception):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


def departure(flight) -> datetime:
    return datetime.fromisoformat(f"{flight['date']}T{flight['dep_time']}")


def window(dep: datetime, at: datetime) -> str:
    hours = (dep - at).total_seconds() / 3600
    if hours < 0:
        return "after"
    if hours < 4:
        return "lt4"
    if hours < 48:
        return "h4"
    if hours < 168:
        return "h48"
    return "d7"


def involuntary_reason(flight):
    if flight["status"] == "cancelled":
        return "GEN-CANCEL"
    if flight["status"] == "delayed" and flight["delay_min"] >= 120:
        return "GEN-DELAY"
    return None


def calc(order_id: str, action: str, target_date: str | None = None, target_flight: str | None = None) -> dict:
    order = data.get_order(order_id)
    if not order:
        raise EngineError("ORDER_NOT_FOUND", f"订单不存在：{order_id}")
    if order["status"] != "ISSUED":
        raise EngineError("ORDER_NOT_ISSUED", f"订单状态为 {order['status']}，不能办理退改")
    flight = data.get_flight(order["flight_no"], order["date"])
    now = S.now()
    w = window(departure(flight), now)
    inv = involuntary_reason(flight)
    rates = data.get_fee_rule(order["airline"], order["fare_product"])
    fare, tax = order["fare"], order["tax"]
    base = {"order_id": order_id, "action": action, "currency": "CNY", "fare": fare, "tax": tax,
            "window": w, "window_label": seed.WINDOW_LABEL[w], "involuntary": bool(inv),
            "flight_status": flight["status"], "delay_min": flight["delay_min"], "allowed": True,
            "reason": "", "rate_pct": 0}

    if action == "CHANGE":
        if not target_date:
            raise EngineError("MISSING_TARGET_DATE", "改签必须提供 target_date（YYYY-MM-DD）")
        tf = data.get_flight(target_flight or order["flight_no"], target_date)
        if not tf or (tf["origin"], tf["dest"]) != (flight["origin"], flight["dest"]):
            raise EngineError("NO_TARGET_FLIGHT", f"{target_date} 没有可改签的航班")
        if tf["status"] == "cancelled" or departure(tf) <= now:
            raise EngineError("TARGET_UNAVAILABLE", f"{tf['flight_no']} {target_date} 不可改签")
        target = {"flight_no": tf["flight_no"], "date": target_date, "dep_time": tf["dep_time"],
                  "fare": tf["fares"][order["fare_product"]]}
        base.update(target=target, change_fee=0, fare_diff=0, total=0)
        if inv:
            return {**base, "rule_id": inv, "reason": "非自愿改签，免改期费、免差价"}
        rule_id = f"{order['airline']}-{order['fare_product']}-CHG"
        if rates["change"] is None:
            return {**base, "rule_id": rule_id, "allowed": False, "reason": "该票价产品不得自愿改签"}
        pct = rates["change"][seed.WINDOWS.index(w)]
        fee = round(fare * pct / 100)
        diff = max(0, target["fare"] - fare)
        return {**base, "rule_id": rule_id, "rate_pct": pct, "change_fee": fee, "fare_diff": diff,
                "total": fee + diff}

    if action == "REFUND":
        base.update(refund_fee=0, refund_amount=0)
        if inv:
            return {**base, "rule_id": inv, "refund_amount": fare + tax, "reason": "非自愿退票，全额退款"}
        rule_id = f"{order['airline']}-{order['fare_product']}-RFD"
        if rates["refund"] is None:
            return {**base, "rule_id": rule_id, "allowed": False, "reason": "该票价产品不得自愿退票"}
        pct = rates["refund"][seed.WINDOWS.index(w)]
        fee = round(fare * pct / 100)
        return {**base, "rule_id": rule_id, "rate_pct": pct, "refund_fee": fee,
                "refund_amount": fare - fee + tax}

    raise EngineError("BAD_ACTION", f"不支持的操作：{action}")
```

- [ ] **Step 4：运行测试，确认通过**

Run: `uv run pytest tests/test_engine.py -v`
Expected: 8 passed

- [ ] **Step 5：提交**

```bash
git add air_agent/engine.py tests/test_engine.py
git commit -m "feat: add fare engine as single source of amounts"
```

### Task 4：状态模型与工具 Schema 校验

**Files:**
- Create: `air_agent/state.py`、`air_agent/tools.py`
- Test: `tests/test_tools.py`

工具参数错误分 3 类，和 work-items 里的失败拆解对应：`MISSING`（缺参）、`TYPE`（类型或格式错误）、`ENUM`（枚举越界）。业务错误（引擎报错、查不到）记为 `ENGINE`。

- [ ] **Step 1：写失败测试 `tests/test_tools.py`**

```python
from datetime import date, timedelta
from air_agent import data, tools


def test_missing_target_date_for_change():
    data.init()
    oid = data.list_orders()[0]["order_id"]
    args, errs = tools.validate("tool_recalc_fee", {"order_id": oid, "action": "CHANGE"})
    assert args is None
    assert errs[0]["field"] == "target_date" and errs[0]["kind"] == "MISSING"


def test_relative_date_rejected_as_type_error():
    args, errs = tools.validate("tool_recalc_fee",
                                {"order_id": "ORD20260926001", "action": "CHANGE", "target_date": "后天"})
    assert errs[0]["field"] == "target_date" and errs[0]["kind"] == "TYPE"
    assert "YYYY-MM-DD" in errs[0]["expected"]


def test_enum_error():
    _, errs = tools.validate("tool_rule_retrieve", {"query": "退票", "cabin": "经济舱"})
    assert errs[0]["kind"] == "ENUM" and errs[0]["field"] == "cabin"


def test_bad_order_id_format():
    _, errs = tools.validate("tool_recalc_fee",
                             {"order_id": "https://x/ORD1", "action": "REFUND"})
    assert errs[0]["kind"] == "TYPE"


def test_unknown_tool():
    _, errs = tools.validate("tool_magic", {})
    assert errs[0]["kind"] == "UNKNOWN_TOOL"


def test_run_recalc_fee_ok_and_engine_error():
    data.init()
    o = data.list_orders("u_zhang")[0]
    ok = tools.run_tool("tool_recalc_fee", {"order_id": o["order_id"], "action": "REFUND"})
    assert ok["ok"] and ok["output"]["refund_amount"] > 0
    bad = tools.run_tool("tool_recalc_fee", {"order_id": "ORD99999999999", "action": "REFUND"})
    assert not bad["ok"] and bad["kind"] == "ENGINE"


def test_run_flight_status():
    data.init()
    o = data.list_orders("u_wang")[0]
    r = tools.run_tool("tool_flight_status", {"flight_no": o["flight_no"], "date": o["date"]})
    assert r["ok"] and r["output"]["status"] in {"scheduled", "delayed", "cancelled"}


def test_llm_specs_cover_react_tools():
    names = [t["function"]["name"] for t in tools.llm_specs()]
    assert names == ["tool_flight_status", "tool_rule_retrieve", "tool_recalc_fee", "finish"]
```

- [ ] **Step 2：运行测试，确认失败**

Run: `uv run pytest tests/test_tools.py -v`
Expected: FAIL，`ImportError: cannot import name 'tools'`

- [ ] **Step 3：实现 `air_agent/state.py`**

```python
"""LangGraph 状态与各模块之间传递的数据模型。"""
import operator
from typing import Annotated, Literal, TypedDict
from pydantic import BaseModel, Field

Intent = Literal["CHANGE", "REFUND", "POLICY", "STATUS"]
INTENT_LABEL = {"CHANGE": "改签", "REFUND": "退票", "POLICY": "政策咨询", "STATUS": "航班状态"}


class SubQuery(BaseModel):
    intent: Intent
    text: str
    target_date: str | None = None


class ParsedQuery(BaseModel):
    intents: list[Intent] = Field(default_factory=list)
    sub_queries: list[SubQuery] = Field(default_factory=list)
    order_id: str | None = None
    airline: str | None = None
    cabin: str | None = None
    fare_product: str | None = None
    scenario: str | None = None
    missing: list[str] = Field(default_factory=list)
    rewritten: str = ""
    step_back: str | None = None
    confidence: float = 1.0
    clarify: dict | None = None      # {"field", "question", "options": [{"label", "value"}]}


class FeeItem(BaseModel):
    name: str
    amount: int


class SchemeOption(BaseModel):
    type: Intent
    title: str
    allowed: bool = True
    fee_items: list[FeeItem] = Field(default_factory=list)
    total: int | None = None
    rule_refs: list[str] = Field(default_factory=list)
    notes: str = ""


class SchemeOutput(BaseModel):
    options: list[SchemeOption]
    text: str


class AgentState(TypedDict, total=False):
    # 输入
    session_id: str
    user_id: str
    order_id: str | None
    question: str
    mode: str                     # offline | deepseek
    # 记忆与理解
    context: dict                 # memory_node 注入的上下文与 Token 预算
    parsed: dict                  # ParsedQuery
    # ReAct
    sub_idx: int
    calls: list[dict]             # 当前子诉求的工具调用记录
    results: list[dict]           # 每个子诉求的结果：{intent, status, rules, funnel, fee}
    next_action: dict             # {thought, action, args, source}
    error_feedback: list[dict]
    rounds: int
    tool_retries: int
    gate_reason: str
    # 生成与校验
    scheme: dict
    verify: dict
    retries: int
    feedback: list[str]
    # 结果
    outcome: str                  # ANSWER | CLARIFY | HUMAN | TICKETED | CANCELLED
    final: dict
    trace: Annotated[list[dict], operator.add]
```

- [ ] **Step 4：实现 `air_agent/tools.py`**

```python
"""Function Calling 工具：参数 Schema、校验与统一执行入口。"""
from typing import Literal
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator
from pydantic_core import PydanticCustomError
from . import data, engine

DATE = r"^\d{4}-\d{2}-\d{2}$"
FLIGHT = r"^[A-Z0-9]{2}\d{3,4}$"
ORDER = r"^ORD\d{11}$"
Cabin = Literal["F", "C", "Y"]
FareProduct = Literal["F", "C", "Y", "Y_DISCOUNT", "SPECIAL", "GROUP"]
Scenario = Literal["VOLUNTARY_CHANGE", "VOLUNTARY_REFUND", "CANCEL", "DELAY", "NO_SHOW", "FARE_DIFF"]


class _Args(BaseModel):
    model_config = ConfigDict(extra="forbid")


class FlightStatusArgs(_Args):
    """查询航班实时状态（正常/延误/取消），用于判断是否属于非自愿退改。"""
    flight_no: str = Field(pattern=FLIGHT, description="航班号，如 MU5101")
    date: str = Field(pattern=DATE, description="航班日期，YYYY-MM-DD 绝对日期")


class RuleRetrieveArgs(_Args):
    """检索退改签规则。"""
    query: str = Field(min_length=1, max_length=200, description="检索问题")
    airline: str | None = Field(default=None, pattern=r"^[A-Z0-9]{2}$", description="航司二字码")
    cabin: Cabin | None = None
    fare_product: FareProduct | None = None
    scenario: Scenario | None = None


class RecalcFeeArgs(_Args):
    """调用运价引擎计算改签费/差价或退票费。金额只能来自这个工具。"""
    order_id: str = Field(pattern=ORDER, description="订单号，ORD + 11 位数字")
    action: Literal["CHANGE", "REFUND"]
    target_date: str | None = Field(default=None, pattern=DATE, description="改签目标日期，YYYY-MM-DD 绝对日期")
    target_flight: str | None = Field(default=None, pattern=FLIGHT)

    @model_validator(mode="after")
    def _need_date(self):
        if self.action == "CHANGE" and not self.target_date:
            raise PydanticCustomError("missing_target_date", "改签必须提供 target_date")
        return self


class FinishArgs(_Args):
    """当前子诉求所需信息已齐全，进入方案生成。"""
    reason: str = Field(default="", max_length=200)


class IssueTicketArgs(_Args):
    order_id: str = Field(pattern=ORDER)
    action: Literal["CHANGE", "REFUND"]
    confirm_id: str = Field(pattern=r"^[\w-]{6,64}$")
    target_date: str | None = Field(default=None, pattern=DATE)
    target_flight: str | None = Field(default=None, pattern=FLIGHT)


SCHEMAS = {"tool_flight_status": FlightStatusArgs, "tool_rule_retrieve": RuleRetrieveArgs,
           "tool_recalc_fee": RecalcFeeArgs, "finish": FinishArgs, "tool_issue_ticket": IssueTicketArgs}
REACT_TOOLS = ["tool_flight_status", "tool_rule_retrieve", "tool_recalc_fee", "finish"]
EXPECTED = {DATE: "YYYY-MM-DD 绝对日期（先把「后天」等相对日期换算好）", FLIGHT: "航班号，如 MU5101",
            ORDER: "订单号，ORD + 11 位数字，不要传链接"}


def _kind(err) -> str:
    t = err["type"]
    if t.startswith("missing"):
        return "MISSING"
    if t in {"literal_error", "enum"}:
        return "ENUM"
    return "TYPE"


def _expected(err) -> str:
    ctx = err.get("ctx") or {}
    if "pattern" in ctx:
        return EXPECTED.get(ctx["pattern"], f"匹配 {ctx['pattern']}")
    if "expected" in ctx:
        return f"取值之一：{ctx['expected']}"
    if err["type"] == "missing_target_date":
        return EXPECTED[DATE]
    return err["msg"]


def validate(tool: str, args: dict):
    """返回 (校验后的模型, 错误列表)。错误项：{field, kind, expected, got, message}。"""
    schema = SCHEMAS.get(tool)
    if schema is None:
        return None, [{"field": "tool", "kind": "UNKNOWN_TOOL", "expected": "、".join(REACT_TOOLS),
                       "got": tool, "message": f"没有工具 {tool}"}]
    try:
        return schema.model_validate(args or {}), []
    except ValidationError as e:
        errs = []
        for err in e.errors():
            field = ".".join(str(x) for x in err["loc"]) or ("target_date" if "target_date" in err["msg"] else "args")
            errs.append({"field": field, "kind": _kind(err), "expected": _expected(err),
                         "got": err.get("input") if err["type"] != "missing" else None,
                         "message": f"{field}：{_expected(err)}"})
        return None, errs


def run_tool(tool: str, args: dict, boost: dict | None = None) -> dict:
    """统一执行入口：{ok, output} 或 {ok: False, kind, errors|error}。"""
    model, errs = validate(tool, args)
    if errs:
        return {"ok": False, "kind": errs[0]["kind"], "errors": errs}
    a = model.model_dump()
    try:
        if tool == "tool_flight_status":
            f = data.get_flight(a["flight_no"], a["date"])
            if not f:
                return {"ok": False, "kind": "ENGINE", "error": {"code": "FLIGHT_NOT_FOUND",
                                                                  "message": f"查不到航班 {a['flight_no']} {a['date']}"}}
            return {"ok": True, "output": {k: f[k] for k in ("flight_no", "date", "dep_time", "status", "delay_min")}}
        if tool == "tool_rule_retrieve":
            from . import retrieval
            filters = {k: a[k] for k in ("airline", "cabin", "fare_product", "scenario") if a[k]}
            return {"ok": True, "output": retrieval.search(a["query"], filters, boost or {})}
        if tool == "tool_recalc_fee":
            return {"ok": True, "output": engine.calc(a["order_id"], a["action"], a["target_date"], a["target_flight"])}
        if tool == "finish":
            return {"ok": True, "output": {"reason": a["reason"]}}
    except engine.EngineError as e:
        return {"ok": False, "kind": "ENGINE", "error": {"code": e.code, "message": str(e)}}
    return {"ok": False, "kind": "UNKNOWN_TOOL", "errors": [{"field": "tool", "message": tool}]}


def llm_specs() -> list[dict]:
    """给 LLM 的 OpenAI 格式工具定义。"""
    out = []
    for name in REACT_TOOLS:
        schema = SCHEMAS[name]
        out.append({"type": "function", "function": {"name": name, "description": schema.__doc__.strip(),
                                                      "parameters": schema.model_json_schema()}})
    return out
```

- [ ] **Step 5：运行测试，确认通过**

Run: `uv run pytest tests/test_tools.py -v`
Expected: 8 passed

- [ ] **Step 6：提交**

```bash
git add air_agent/state.py air_agent/tools.py tests/test_tools.py
git commit -m "feat: add agent state models and validated tools"
```

### Task 5：混合检索漏斗（工作三）

**Files:**
- Create: `air_agent/retrieval.py`
- Test: `tests/test_retrieval.py`、`tests/test_retrieval_models.py`（slow）

**精排得分** = 0.7 × 归一化相关性 + 0.3 × 字段匹配度。
- 归一化相关性：hybrid 模式取 Cross-Encoder 分数，fallback 模式取 BM25 分数，都做 min-max 归一化。
- 字段匹配度：航司、票价产品各占权重 1，完全匹配记 1 分，通用规则 `*` 记 0.5 分；场景占权重 2，匹配记满分。
- 低于 `rerank_min` 的结果截掉（动态截断）。
- 硬过滤层同时做绝对相关性门槛：候选必须命中 BM25，或者向量相似度 ≥ `dense_min`。这样完全无关的问题（如「宠物能带上飞机吗」）会返回空结果，后续由校验转人工，避免 min-max 归一化后总有一条「第一名」被硬塞给生成。

- [ ] **Step 1：写失败测试 `tests/test_retrieval.py`**

```python
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
```

- [ ] **Step 2：写 `tests/test_retrieval_models.py`（真实模型，默认跳过）**

```python
import pytest
from air_agent import data, retrieval
from air_agent import settings as S

pytestmark = pytest.mark.slow


def test_hybrid_backend_with_real_models():
    S.update({"retrieval": {"backend": "hybrid"}})
    data.init()
    retrieval.invalidate()
    r = retrieval.search("打折票改到后天要交多少改期费", {"airline": "MU", "fare_product": "Y_DISCOUNT",
                                                    "scenario": "VOLUNTARY_CHANGE"})
    assert r["backend"] == "hybrid" and not r["degraded"]
    assert r["rules"][0]["id"] == "MU-Y_DISCOUNT-CHG"
```

- [ ] **Step 3：运行测试，确认失败**

Run: `uv run pytest tests/test_retrieval.py -v`
Expected: FAIL，`ImportError: cannot import name 'retrieval'`

- [ ] **Step 4：实现 `air_agent/retrieval.py`**

```python
"""工作三：向量 + BM25 双路召回 → RRF 融合 → 硬过滤 → 精排（动态截断）→ MMR → 注入。每层留痕。"""
import logging
import math
import os
import re
import threading
from collections import Counter
from . import data
from . import settings as S

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
MODEL_DIR = S.ROOT / "models"
EMB_MODEL = "BAAI/bge-small-zh-v1.5"
RERANK_MODEL = "ms-marco-MultiBERT-L-12"
CODE = re.compile(r"[A-Z][A-Z0-9_]+|\d+%?")
STAGE_LABEL = {"dense": "向量召回", "sparse": "BM25 召回", "rrf": "RRF 融合", "filter": "硬过滤",
               "rerank": "精排", "mmr": "MMR 去冗余", "inject": "注入生成"}

_lock = threading.Lock()
_models = {"loaded": False, "embed": None, "rerank": None, "error": ""}
_index = {"key": None, "value": None}


def tokenize(text: str) -> list[str]:
    import jieba
    jieba.setLogLevel(logging.WARNING)
    words = [w for w in jieba.lcut(text) if w.strip() and not re.fullmatch(r"[\W_]+", w)]
    return words + CODE.findall(text)


def _load_models():
    if _models["loaded"]:
        return
    _models["loaded"] = True
    try:
        from fastembed import TextEmbedding
        from flashrank import Ranker
        _models["embed"] = TextEmbedding(EMB_MODEL, cache_dir=str(MODEL_DIR))
        _models["rerank"] = Ranker(model_name=RERANK_MODEL, cache_dir=str(MODEL_DIR))
    except Exception as e:  # 下载失败、离线等 → 降级
        _models.update(embed=None, rerank=None, error=f"{type(e).__name__}: {e}")


def status() -> dict:
    want = S.get()["retrieval"]["backend"]
    mode = "fallback"
    if want == "hybrid":
        _load_models()
        if _models["embed"] is not None:
            mode = "hybrid"
    return {"configured": want, "backend": mode, "degraded": want == "hybrid" and mode != "hybrid",
            "reason": _models["error"] if want == "hybrid" and mode != "hybrid" else ""}


# ---------- 降级用的字符 n-gram TF-IDF ----------
def _ngrams(t):
    t = re.sub(r"\s+", "", t)
    return [t[i:i + n] for n in (1, 2) for i in range(len(t) - n + 1)]


def _weigh(cnt, idf):
    v = {g: c * idf[g] for g, c in cnt.items() if g in idf}
    n = math.sqrt(sum(x * x for x in v.values())) or 1.0
    return {g: x / n for g, x in v.items()}


def _cos(a, b):
    if isinstance(a, dict):
        return sum(x * b.get(g, 0.0) for g, x in a.items())
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a)) or 1.0
    nb = math.sqrt(sum(y * y for y in b)) or 1.0
    return dot / (na * nb)


class Index:
    def __init__(self, rules: list[dict], mode: str):
        from rank_bm25 import BM25Okapi
        self.rules, self.mode = rules, mode
        self.texts = [f"{r['title']}。{r['content']}" for r in rules]
        self.bm25 = BM25Okapi([tokenize(t) for t in self.texts])
        if mode == "hybrid":
            from qdrant_client import QdrantClient, models as qm
            self.vecs = [[float(x) for x in v] for v in _models["embed"].embed(self.texts)]
            self.qdrant = QdrantClient(":memory:")
            self.qdrant.create_collection(
                "rules", vectors_config=qm.VectorParams(size=len(self.vecs[0]), distance=qm.Distance.COSINE),
                hnsw_config=qm.HnswConfigDiff(m=16, ef_construct=100))
            self.qdrant.upsert("rules", points=[qm.PointStruct(id=i, vector=v, payload={"rule_id": r["id"]})
                                                for i, (v, r) in enumerate(zip(self.vecs, rules))])
        else:
            docs = [Counter(_ngrams(t)) for t in self.texts]
            df = Counter(g for d in docs for g in d)
            n = len(docs)
            self.idf = {g: math.log((n + 1) / (c + 1)) + 1 for g, c in df.items()}
            self.vecs = [_weigh(d, self.idf) for d in docs]

    def query_vec(self, q):
        if self.mode == "hybrid":
            return [float(x) for x in next(iter(_models["embed"].query_embed(q)))]
        return _weigh(Counter(_ngrams(q)), self.idf)

    def dense(self, q, k):
        qv = self.query_vec(q)
        if self.mode == "hybrid":
            hits = self.qdrant.query_points("rules", query=qv, limit=k).points
            return [(h.id, float(h.score)) for h in hits]
        sims = [(i, _cos(qv, v)) for i, v in enumerate(self.vecs)]
        return sorted([x for x in sims if x[1] > 0], key=lambda x: -x[1])[:k]

    def sparse(self, q, k):
        scores = self.bm25.get_scores(tokenize(q))
        return sorted([(i, float(s)) for i, s in enumerate(scores) if s > 0], key=lambda x: -x[1])[:k]


def invalidate():
    with _lock:
        _index.update(key=None, value=None)


def get_index() -> Index:
    rules = data.list_rules()
    mode = status()["backend"]
    key = (mode, len(rules), max((r["updated_at"] for r in rules), default=""), tuple(r["id"] for r in rules))
    with _lock:
        if _index["key"] != key:
            _index.update(key=key, value=Index(rules, mode))
        return _index["value"]


def _match(rule, filters):
    for f in ("airline", "cabin", "fare_product"):
        if filters.get(f) and rule[f] not in (filters[f], "*"):
            return False
    return True


def _field_score(rule, filters):
    got, total = 0.0, 0.0
    for f in ("airline", "fare_product"):
        if filters.get(f):
            total += 1
            got += 1.0 if rule[f] == filters[f] else 0.5
    if filters.get("scenario"):
        total += 2
        got += 2.0 if rule["scenario"] == filters["scenario"] else 0.0
    return got / total if total else None


def _rerank(idx, query, cands, filters, boost, cfg):
    if not cands:
        return []
    if idx.mode == "hybrid":
        from flashrank import RerankRequest
        res = _models["rerank"].rerank(RerankRequest(query=query, passages=[{"id": i, "text": idx.texts[i]}
                                                                            for i, _ in cands]))
        raw = {int(r["id"]): float(r["score"]) for r in res}
    else:
        bm = idx.bm25.get_scores(tokenize(query))
        qv = idx.query_vec(query)
        raw = {i: float(bm[i]) + 5 * _cos(qv, idx.vecs[i]) for i, _ in cands}
    lo, hi = min(raw.values()), max(raw.values())
    out = []
    for i, _ in cands:
        rel = (raw[i] - lo) / (hi - lo) if hi > lo else 1.0
        fs = _field_score(idx.rules[i], filters)
        score = rel if fs is None else 0.7 * rel + 0.3 * fs
        r = idx.rules[i]
        if r["airline"] in boost.get("airlines", []) or r["fare_product"] in boost.get("products", []):
            score *= 1 + cfg["profile_boost"]
        out.append((i, score))
    return sorted(out, key=lambda x: -x[1])


def _mmr(idx, cands, lam, k):
    chosen, pool = [], list(cands)
    while pool and len(chosen) < k:
        def gain(c):
            div = max((_cos(idx.vecs[c[0]], idx.vecs[s[0]]) for s in chosen), default=0.0)
            return lam * c[1] - (1 - lam) * div
        best = max(pool, key=gain)
        chosen.append(best)
        pool.remove(best)
    return chosen


def search(query: str, filters: dict | None = None, boost: dict | None = None) -> dict:
    cfg = S.get()["retrieval"]
    filters, boost = filters or {}, boost or {}
    idx = get_index()
    funnel = []

    def stage(name, k, items):
        funnel.append({"stage": name, "label": STAGE_LABEL[name], "k": k, "count": len(items),
                       "items": [{"id": idx.rules[i]["id"], "title": idx.rules[i]["title"], "score": round(s, 4)}
                                 for i, s in items]})

    dense = idx.dense(query, cfg["k_recall"])
    stage("dense", cfg["k_recall"], dense)
    sparse = idx.sparse(query, cfg["k_recall"])
    stage("sparse", cfg["k_recall"], sparse)
    rrf = Counter()
    for ranked in (dense, sparse):
        for rank, (i, _) in enumerate(ranked):
            rrf[i] += 1 / (cfg["rrf_k"] + rank + 1)
    fused = rrf.most_common(cfg["k_fuse"])
    stage("rrf", cfg["k_fuse"], fused)
    sparse_ids = {i for i, _ in sparse}
    dense_score = dict(dense)
    kept = [(i, s) for i, s in fused if _match(idx.rules[i], filters)
            and (i in sparse_ids or dense_score.get(i, 0) >= cfg["dense_min"][idx.mode])]   # 绝对相关性门槛
    stage("filter", None, kept)
    ranked = _rerank(idx, query, kept, filters, boost, cfg)
    top = [x for x in ranked if x[1] >= cfg["rerank_min"]][:cfg["k_rerank"]]
    stage("rerank", cfg["k_rerank"], top)
    diverse = _mmr(idx, top, cfg["mmr_lambda"], cfg["k_mmr"])
    stage("mmr", cfg["k_mmr"], diverse)
    inject = diverse[:cfg["k_inject"]]
    stage("inject", cfg["k_inject"], inject)
    st = status()
    return {"query": query, "filters": filters,
            "rules": [{**{k: idx.rules[i][k] for k in ("id", "airline", "cabin", "fare_product", "scenario",
                                                        "title", "content")}, "score": round(s, 4)}
                      for i, s in inject],
            "funnel": funnel, "backend": idx.mode, "degraded": st["degraded"], "degraded_reason": st["reason"]}
```

- [ ] **Step 5：运行测试，确认通过**

Run: `uv run pytest tests/test_retrieval.py -v`
Expected: 7 passed。如果 `test_off_topic_returns_nothing` 失败，打印 dense 层得分，把 `dense_min.fallback` 调到无关问题最高分之上、相关问题最低分之下，并在提交信息里写明取值。如果 `test_profile_boost_changes_score` 失败，说明 HU 规则的归一化分数和其他航司差得太多；先打印 `boosted["funnel"]` 里 rerank 层的得分，再判断是测试的查询词需要调，还是 boost 计算有误。不要直接放宽断言。

- [ ] **Step 6：用真实模型跑一次 slow 测试**

Run: `uv run pytest tests/test_retrieval_models.py -m slow -v`
Expected: 1 passed。首次运行会从 hf-mirror 下载约 100MB 模型到 `models/`。

- [ ] **Step 7：提交**

```bash
git add air_agent/retrieval.py tests/test_retrieval.py tests/test_retrieval_models.py
git commit -m "feat: add hybrid retrieval funnel with fallback"
```

### Task 6：Query 理解（工作二）

**Files:**
- Create: `air_agent/query.py`
- Test: `tests/test_query.py`

本任务只做规则模式。DeepSeek 模式在 Task 9 的 `llm.py` 里实现，调用 `understand(..., llm=client)`：模型只负责拆诉求和抽日期，订单字段对齐仍走规则。

测试固定「今天」为 2026-09-26（周六），所以「大后天」是 09-29，「下周六」是 10-03。

- [ ] **Step 1：写失败测试 `tests/test_query.py`**

```python
from air_agent import data, query

D = "20260926"


def ctx(**kw):
    base = {"user_id": "u_zhang", "selected_order_id": None, "order_stack": [], "last_intent": None}
    return {**base, **kw}


def setup_module(_):
    data.init()


def test_resolve_dates():
    assert query.resolve_date("大后天") == "2026-09-29"
    assert query.resolve_date("下周六") == "2026-10-03"
    assert query.resolve_date("10月3号") == "2026-10-03"
    assert query.resolve_date("2026-10-05") == "2026-10-05"
    assert query.resolve_date("下个月初") is None


def test_multi_intent_split_and_dates():
    p = query.understand("帮我改到大后天，顺便看看退票多少钱", ctx(selected_order_id=f"ORD{D}001"))
    assert [s.intent for s in p.sub_queries] == ["CHANGE", "REFUND"]
    assert p.sub_queries[0].target_date == "2026-09-29"
    assert p.order_id == f"ORD{D}001" and p.airline == "MU" and p.fare_product == "Y_DISCOUNT"
    assert p.clarify is None and p.confidence >= 0.85


def test_return_leg_coreference():
    p = query.understand("回程那张能退吗", ctx(order_stack=[f"ORD{D}001"]))
    assert p.order_id == f"ORD{D}002"


def test_pronoun_uses_stack_top():
    p = query.understand("它退票要多少钱", ctx(order_stack=[f"ORD{D}003", f"ORD{D}001"]))
    assert p.order_id == f"ORD{D}003"


def test_ambiguous_order_asks_user():
    p = query.understand("我要退票", ctx())
    assert p.clarify["field"] == "order_id"
    assert len(p.clarify["options"]) >= 2
    assert p.confidence < 0.85


def test_change_without_date_asks_date():
    p = query.understand("我想改签", ctx(selected_order_id=f"ORD{D}001"))
    assert p.clarify["field"] == "target_date"
    assert p.clarify["options"][0]["value"] == "2026-09-27"


def test_unparseable_date_kept_raw():
    p = query.understand("改到下个月初", ctx(selected_order_id=f"ORD{D}001"))
    assert p.sub_queries[0].target_date == "下个月初" and p.clarify is None


def test_ellipsis_inherits_last_intent():
    p = query.understand("那大后天呢", ctx(selected_order_id=f"ORD{D}001", last_intent="CHANGE"))
    assert p.sub_queries[0].intent == "CHANGE" and p.sub_queries[0].target_date == "2026-09-29"


def test_policy_without_order_uses_synonyms():
    p = query.understand("东方云航特价票能退吗", ctx(user_id="u_wang_nobody"))
    assert p.sub_queries[0].intent == "POLICY"
    assert p.airline == "MU" and p.fare_product == "SPECIAL" and p.scenario == "VOLUNTARY_REFUND"


def test_scenario_from_flight_status():
    cancelled = next(o for o in data.list_orders("u_wang")
                     if data.get_flight(o["flight_no"], o["date"])["status"] == "cancelled")
    p = query.understand("航班取消了怎么办，我要退票", ctx(user_id="u_wang", selected_order_id=cancelled["order_id"]))
    assert p.scenario == "CANCEL"


def test_clarified_values_override():
    p = query.understand("改到下个月初", ctx(selected_order_id=f"ORD{D}001", forced_date="2026-10-02"))
    assert p.sub_queries[0].target_date == "2026-10-02"
    p = query.understand("嗯", ctx(selected_order_id=f"ORD{D}001", forced_intent="REFUND"))
    assert p.sub_queries[0].intent == "REFUND" and p.clarify is None


def test_step_back_and_hyde_text():
    p = query.understand("改到大后天", ctx(selected_order_id=f"ORD{D}001"))
    assert query.step_back_query(p, p.sub_queries[0]) == "经济舱自愿改签规则"
    assert "改期" in query.hyde_text(p, p.sub_queries[0])
```

- [ ] **Step 2：运行测试，确认失败**

Run: `uv run pytest tests/test_query.py -v`
Expected: FAIL，`ImportError: cannot import name 'query'`

- [ ] **Step 3：实现 `air_agent/query.py`**

```python
"""工作二：订单字段对齐、意图与多诉求拆解、指代消解、日期换算、Step-back、HyDE。"""
import re
from datetime import date, timedelta
from . import data, engine, seed
from . import settings as S
from .state import INTENT_LABEL, ParsedQuery, SubQuery

INTENT_PATTERNS = [
    ("CHANGE", r"改签|改到|改成|换到|改期|改一下|换个时间|延后|提前到|往后挪"),
    ("REFUND", r"退票|退掉|退了|退款|不坐了|不去了|退多少|能退"),
    ("STATUS", r"航班状态|延误了吗|取消了吗|正常起飞|准点|啥情况|什么情况|还飞吗"),
    ("POLICY", r"规则|政策|规定"),
]
SPLIT = r"顺便|另外|还有|同时|以及|并且|再帮我|再看"
WEEK = "一二三四五六日"
DATE_PHRASE = (r"\d{4}-\d{2}-\d{2}|大后天|后天|明天|今天|下(?:周|星期)[一二三四五六日天]|"
               r"(?:周|星期)[一二三四五六日天]|\d{1,2}月\d{1,2}[号日]?|\d{1,2}[号日]")
VAGUE_DATE = r"下个?月初|下个?月底|月初|月底|过几天|过两天|节后|年后|下下周[一二三四五六日天]?"
PRODUCT_SYNONYMS = [("SPECIAL", r"特价"), ("GROUP", r"团队|团购"), ("Y_DISCOUNT", r"打折|折扣"),
                    ("F", r"头等"), ("C", r"公务|商务舱"), ("Y", r"全价|经济舱")]
AIRLINE_SYNONYMS = {code: name for code, name in seed.AIRLINES.items()}
CABIN_LABEL = {"F": "头等舱", "C": "公务舱", "Y": "经济舱"}
SCENARIO_KIND = {"CHANGE": "VOLUNTARY_CHANGE", "REFUND": "VOLUNTARY_REFUND"}
KIND_LABEL = {"VOLUNTARY_CHANGE": "自愿改签", "VOLUNTARY_REFUND": "自愿退票", "CANCEL": "航班取消",
              "DELAY": "航班延误", "NO_SHOW": "误机", "FARE_DIFF": "改签差价"}


def today() -> date:
    return S.now().date()


def resolve_date(text: str) -> str | None:
    t, d0 = text.strip(), today()
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", t):
        return t
    rel = {"今天": 0, "明天": 1, "后天": 2, "大后天": 3}
    if t in rel:
        return (d0 + timedelta(days=rel[t])).isoformat()
    m = re.fullmatch(r"(下)?(?:周|星期)([一二三四五六日天])", t)
    if m:
        wd = WEEK.index(m.group(2).replace("天", "日"))
        if m.group(1):
            return (d0 + timedelta(days=7 - d0.weekday() + wd)).isoformat()
        delta = wd - d0.weekday()
        return (d0 + timedelta(days=delta if delta > 0 else delta + 7)).isoformat()
    m = re.fullmatch(r"(\d{1,2})月(\d{1,2})[号日]?", t)
    if m:
        y = d0.year if (int(m.group(1)), int(m.group(2))) >= (d0.month, d0.day) else d0.year + 1
        return date(y, int(m.group(1)), int(m.group(2))).isoformat()
    m = re.fullmatch(r"(\d{1,2})[号日]", t)
    if m:
        day = int(m.group(1))
        y, mo = (d0.year, d0.month) if day >= d0.day else ((d0.year + 1, 1) if d0.month == 12 else (d0.year, d0.month + 1))
        return date(y, mo, day).isoformat()
    return None


def find_date(text: str):
    """返回 (标准日期 | 原始模糊短语 | None)。"""
    m = re.search(DATE_PHRASE, text)
    if m:
        return resolve_date(m.group(0)) or m.group(0)
    m = re.search(VAGUE_DATE, text)
    return m.group(0) if m else None


def detect_intents(text: str) -> list[str]:
    hits = []
    for intent, pat in INTENT_PATTERNS:
        m = re.search(pat, text)
        if m:
            hits.append((m.start(), intent))
    return [i for _, i in sorted(hits)]


def order_label(o: dict) -> str:
    f = data.get_flight(o["flight_no"], o["date"])
    return (f"{o['flight_no']} {f['origin_city']}→{f['dest_city']} {o['date']} {f['dep_time']} "
            f"{seed.PRODUCTS[o['fare_product']][1]}")


def _future_orders(user_id):
    now = S.now()
    return [o for o in data.list_orders(user_id) if o["status"] == "ISSUED"
            and engine.departure(data.get_flight(o["flight_no"], o["date"])) > now]


def resolve_order(text: str, ctx: dict) -> str | None:
    orders = data.list_orders(ctx["user_id"])
    by_id = {o["order_id"]: o for o in orders}
    m = re.search(r"ORD\d{11}", text)
    if m and m.group(0) in by_id:
        return m.group(0)
    stack = [x for x in ctx.get("order_stack", []) if x in by_id]
    anchor = ctx.get("selected_order_id") or (stack[0] if stack else None)
    if re.search(r"回程|返程|回来那", text) and anchor:
        a = data.get_flight(by_id[anchor]["flight_no"], by_id[anchor]["date"])
        for o in orders:
            f = data.get_flight(o["flight_no"], o["date"])
            if (f["origin"], f["dest"]) == (a["dest"], a["origin"]) and o["order_id"] != anchor:
                return o["order_id"]
    if re.search(r"上一张|之前那张|刚才那张|另一张", text) and len(stack) > 1:
        return stack[1]
    m = re.search(r"[A-Z0-9]{2}\d{3,4}", text)
    if m:
        for o in orders:
            if o["flight_no"] == m.group(0):
                return o["order_id"]
    for o in orders:
        f = data.get_flight(o["flight_no"], o["date"])
        if re.search(f"去{f['dest_city']}|{f['origin_city']}飞{f['dest_city']}|{f['origin_city']}到{f['dest_city']}", text):
            return o["order_id"]
    if ctx.get("selected_order_id") in by_id:
        return ctx["selected_order_id"]
    if stack and re.search(r"它|这张|这个|那张|这班", text):
        return stack[0]
    future = _future_orders(ctx["user_id"])
    if len(future) == 1:
        return future[0]["order_id"]
    return stack[0] if stack else None


def _flight_scenario(order) -> str | None:
    f = data.get_flight(order["flight_no"], order["date"])
    inv = engine.involuntary_reason(f)
    if inv:
        return {"GEN-CANCEL": "CANCEL", "GEN-DELAY": "DELAY"}[inv]
    if engine.departure(f) <= S.now():
        return "NO_SHOW"
    return None


def date_options(n=5):
    d0 = today()
    return [{"label": f"{(d0 + timedelta(days=i)).isoformat()}（{'周' + WEEK[(d0 + timedelta(days=i)).weekday()]}）",
             "value": (d0 + timedelta(days=i)).isoformat()} for i in range(1, n + 1)]


def understand(question: str, ctx: dict, llm=None) -> ParsedQuery:
    segments = [s for s in re.split(SPLIT, question) if s.strip()] or [question]
    subs, guessed = [], False
    for seg in segments:
        for intent in detect_intents(seg):
            if not any(s.intent == intent for s in subs):
                subs.append(SubQuery(intent=intent, text=seg.strip(" ，,。"),
                                     target_date=find_date(seg) if intent == "CHANGE" else None))
    if llm is not None:
        subs = llm.parse_intents(question, ctx, subs) or subs
    if not subs:
        intent = ctx.get("forced_intent") or ctx.get("last_intent") or "POLICY"
        guessed = not (ctx.get("forced_intent") or ctx.get("last_intent"))
        subs = [SubQuery(intent=intent, text=question, target_date=find_date(question) if intent == "CHANGE" else None)]
    for s in subs:   # 改签日期可能写在别的分句里；追问得到的日期优先
        if s.intent == "CHANGE":
            s.target_date = ctx.get("forced_date") or s.target_date or find_date(question)

    p = ParsedQuery(intents=[s.intent for s in subs], sub_queries=subs)
    oid = resolve_order(question, ctx)
    if oid:
        o = data.get_order(oid)
        p.order_id, p.airline, p.cabin, p.fare_product = oid, o["airline"], o["cabin"], o["fare_product"]
        p.scenario = _flight_scenario(o)
    else:
        for code, name in AIRLINE_SYNONYMS.items():
            if name in question or re.search(rf"\b{code}\b", question):
                p.airline = code
        for prod, pat in PRODUCT_SYNONYMS:
            if re.search(pat, question):
                p.fare_product, p.cabin = prod, seed.PRODUCTS[prod][0]
                break
        if p.fare_product or p.airline:   # 没有订单但说了票种：按政策咨询处理
            first = subs[0].intent
            p.scenario = SCENARIO_KIND.get(first)
            p.sub_queries = [SubQuery(intent="POLICY", text=question)]
            p.intents = ["POLICY"]
    if re.search(r"取消了|被取消", question) and not p.scenario:
        p.scenario = "CANCEL"
    if re.search(r"延误", question) and not p.scenario:
        p.scenario = "DELAY"

    need_order = any(s.intent in ("CHANGE", "REFUND", "STATUS") for s in p.sub_queries)
    if need_order and not p.order_id:
        p.missing.append("order_id")
    if any(s.intent == "CHANGE" and not s.target_date for s in p.sub_queries):
        p.missing.append("target_date")
    p.confidence = round(max(0.0, 1.0 - 0.3 * ("order_id" in p.missing) - 0.2 * ("target_date" in p.missing)
                             - 0.3 * guessed), 2)

    if "order_id" in p.missing:
        opts = [{"label": order_label(o), "value": o["order_id"]} for o in _future_orders(ctx["user_id"])]
        p.clarify = {"field": "order_id", "question": "您想办理哪一张订单？", "options": opts}
    elif "target_date" in p.missing:
        p.clarify = {"field": "target_date", "question": "想改签到哪一天？", "options": date_options()}
    elif p.confidence < S.get()["query"]["clarify_threshold"]:
        p.clarify = {"field": "intent", "question": "您是想改签、退票，还是了解规则？",
                     "options": [{"label": v, "value": k} for k, v in INTENT_LABEL.items()]}

    desc = f"订单{p.order_id}（{order_label(data.get_order(p.order_id))}）" if p.order_id else "未指定订单"
    asks = "；".join(INTENT_LABEL[s.intent] + (f"至{s.target_date}" if s.target_date else "") for s in p.sub_queries)
    p.rewritten = f"{desc}：{asks}"
    return p


def scenario_for(p: ParsedQuery, sub: SubQuery) -> str | None:
    if p.scenario in ("CANCEL", "DELAY", "NO_SHOW"):
        return p.scenario
    return SCENARIO_KIND.get(sub.intent) or p.scenario


def step_back_query(p: ParsedQuery, sub: SubQuery) -> str:
    cabin = CABIN_LABEL.get(p.cabin or "", "")
    return f"{cabin}{KIND_LABEL.get(scenario_for(p, sub) or '', '退改')}规则"


def hyde_text(p: ParsedQuery, sub: SubQuery) -> str:
    name = seed.AIRLINES.get(p.airline or "", "")
    prod = seed.PRODUCTS.get(p.fare_product or "", ("", ""))[1]
    kind = KIND_LABEL.get(scenario_for(p, sub) or "", "退改")
    verb = "改期" if sub.intent == "CHANGE" else "退票"
    return f"{name}{prod}{kind}规则：起飞前不同时间申请{kind}，按票面价一定比例收取{verb}费。"
```

- [ ] **Step 4：运行测试，确认通过**

Run: `uv run pytest tests/test_query.py -v`
Expected: 12 passed

- [ ] **Step 5：提交**

```bash
git add air_agent/query.py tests/test_query.py
git commit -m "feat: add query understanding (multi-intent, coreference, dates)"
```

### Task 7：上下文与记忆（工作四）

**Files:**
- Create: `air_agent/memory.py`
- Modify: `air_agent/data.py`（SCHEMA 里加 `session_memory` 表）
- Test: `tests/test_memory.py`

- [ ] **Step 1：写失败测试 `tests/test_memory.py`**

```python
from air_agent import data, memory
from air_agent import settings as S

D = "20260926"


def setup_function(_):
    data.init()


def test_order_stack_dedup_and_cap():
    mem = memory.load_session("s1")
    for i in [1, 2, 3, 1, 4, 5, 6]:
        memory.push_order(mem, f"ORD{D}{i:03d}")
    assert mem["order_stack"][0] == f"ORD{D}006"
    assert len(mem["order_stack"]) == 5 and mem["order_stack"].count(f"ORD{D}001") == 1


def test_sliding_window_summary():
    mem = memory.load_session("s2")
    for i in range(6):
        memory.add_turn(mem, "user", f"第{i}轮：改到大后天")
        memory.add_turn(mem, "assistant", f"第{i}轮答复：改期费 100 元")
    assert len(mem["turns"]) == 8
    assert "第0轮" in mem["summary"] and "第5轮" not in mem["summary"]


def test_session_roundtrip():
    mem = memory.load_session("s3")
    memory.push_order(mem, f"ORD{D}001")
    mem["last_intent"] = "CHANGE"
    memory.save_session("s3", "u_zhang", mem)
    again = memory.load_session("s3")
    assert again["order_stack"] == [f"ORD{D}001"] and again["last_intent"] == "CHANGE"


def test_profile_update_is_deduplicated():
    o = data.get_order(f"ORD{D}003")
    before = data.get_profile("u_zhang")["data"]["airlines"].get("CA", 0)
    memory.update_profile("u_zhang", o, "REFUND", key="s4:t1")
    memory.update_profile("u_zhang", o, "REFUND", key="s4:t1")
    prof = data.get_profile("u_zhang")["data"]
    assert prof["airlines"]["CA"] == before + 1
    assert prof["prefer_action"]["REFUND"] == 2


def test_profile_boost():
    assert memory.profile_boost("u_zhang") == {"airlines": ["MU"], "products": ["Y_DISCOUNT"]}
    assert memory.profile_boost("u_wang") == {"airlines": [], "products": []}


def test_clear_long_memory():
    memory.clear_profile("u_zhang")
    assert data.get_profile("u_zhang")["data"]["airlines"] == {}


def test_token_budget():
    S.update({"memory": {"token_budget": 600}})
    mem = memory.load_session("s5")
    for i in range(10):
        memory.add_turn(mem, "user", "我想把杭州飞北京那张票改到大后天，顺便问下退票多少钱" * 2)
    rules = [{"id": f"R{i}", "title": "规则", "content": "起飞前7天以上申请改签收取票面价5%的改期费。" * 6}
             for i in range(10)]
    ctx = memory.build_context(data.get_order(f"ORD{D}001"), rules, mem, data.get_profile("u_zhang"))
    assert ctx["total"] <= 600
    assert ctx["full_total"] > ctx["total"] and ctx["saved_pct"] > 0
    assert ctx["blocks"]["order"]["truncated"] is False
    assert set(ctx["blocks"]) == {"system", "order", "rules", "history", "profile"}
```

- [ ] **Step 2：运行测试，确认失败**

Run: `uv run pytest tests/test_memory.py -v`
Expected: FAIL，`ImportError: cannot import name 'memory'`

- [ ] **Step 3：在 `air_agent/data.py` 的 `SCHEMA` 末尾加一张表**

```sql
CREATE TABLE IF NOT EXISTS session_memory(session_id TEXT PRIMARY KEY, user_id TEXT, data TEXT, updated_at TEXT);
```

- [ ] **Step 4：实现 `air_agent/memory.py`**

```python
"""工作四：短期会话记忆（订单栈 + 滑动窗口摘要）、长期用户画像、Token 预算分配。"""
import json
import math
from . import data, seed
from . import settings as S

SYSTEM_PROMPT = ("你是航空公司退改签助手。只能依据检索到的规则和运价引擎返回的金额作答；"
                 "金额必须引用工具结果，不得自行计算或编造；拿不准时说明需要转人工。")
PRIORITY = ["system", "order", "rules", "history", "profile"]


def tokens(text: str) -> int:
    return math.ceil(len(text) / 1.5)


def _empty():
    return {"order_stack": [], "progress": {}, "turns": [], "summary": "", "last_intent": None}


def load_session(session_id: str) -> dict:
    with data.connect() as c:
        r = c.execute("SELECT data FROM session_memory WHERE session_id=?", (session_id,)).fetchone()
    return {**_empty(), **json.loads(r["data"])} if r else _empty()


def save_session(session_id: str, user_id: str, mem: dict):
    with data.connect() as c:
        c.execute("INSERT OR REPLACE INTO session_memory VALUES(?,?,?,?)",
                  (session_id, user_id, json.dumps(mem, ensure_ascii=False), S.now().isoformat()))


def push_order(mem: dict, order_id: str):
    cap = S.get()["memory"]["order_stack"]
    mem["order_stack"] = ([order_id] + [x for x in mem["order_stack"] if x != order_id])[:cap]


def summarize(turns: list[dict], llm=None) -> str:
    if llm is not None:
        text = llm.summarize(turns)
        if text:
            return text
    return "；".join(f"{'用户' if t['role'] == 'user' else '助手'}：{t['text'][:40]}" for t in turns)


def add_turn(mem: dict, role: str, text: str, llm=None):
    mem["turns"].append({"role": role, "text": text})
    keep = S.get()["memory"]["recent_turns"] * 2
    if len(mem["turns"]) > keep:
        old, mem["turns"] = mem["turns"][:-keep], mem["turns"][-keep:]
        mem.setdefault("archive", []).extend(old)
        piece = summarize(old, llm)
        mem["summary"] = f"{mem['summary']}；{piece}" if mem["summary"] else piece


def _bump(d: dict, key: str):
    d[key] = d.get(key, 0) + 1


def update_profile(user_id: str, order: dict, intent: str, key: str):
    prof = data.get_profile(user_id)
    if not prof or not order:
        return
    p = prof["data"]
    seen = p.setdefault("_seen", [])
    if key in seen:
        return
    seen.append(key)
    p["_seen"] = seen[-200:]
    f = data.get_flight(order["flight_no"], order["date"])
    _bump(p.setdefault("routes", {}), f"{f['origin']}-{f['dest']}")
    _bump(p.setdefault("airlines", {}), order["airline"])
    _bump(p.setdefault("products", {}), order["fare_product"])
    if intent in ("CHANGE", "REFUND"):
        _bump(p.setdefault("prefer_action", {}), intent)
    data.save_profile(user_id, p)


def clear_profile(user_id: str):
    data.save_profile(user_id, {"routes": {}, "airlines": {}, "products": {}, "prefer_action": {},
                                "price_sensitivity": "低"})


def profile_boost(user_id: str) -> dict:
    prof = data.get_profile(user_id)
    if not prof:
        return {"airlines": [], "products": []}
    p = prof["data"]
    return {"airlines": [k for k, v in p.get("airlines", {}).items() if v >= 3],
            "products": [k for k, v in p.get("products", {}).items() if v >= 3]}


def profile_text(prof: dict | None) -> str:
    if not prof:
        return ""
    p = prof["data"]
    top = lambda d: "、".join(k for k, _ in sorted(d.items(), key=lambda x: -x[1])[:3]) or "无"
    return (f"{prof['name']}（{prof['tier']}）常走航线：{top(p.get('routes', {}))}；偏好航司：{top(p.get('airlines', {}))}；"
            f"常买票种：{top(p.get('products', {}))}；退改偏好：{top(p.get('prefer_action', {}))}；"
            f"价格敏感度：{p.get('price_sensitivity', '中')}")


def order_text(order: dict | None) -> str:
    if not order:
        return "未指定订单"
    f = data.get_flight(order["flight_no"], order["date"])
    return (f"订单{order['order_id']}：{seed.AIRLINES[order['airline']]} {order['flight_no']} "
            f"{f['origin_city']}→{f['dest_city']} {order['date']} {f['dep_time']} "
            f"{seed.PRODUCTS[order['fare_product']][1]} 票面价{order['fare']}元 航班状态{f['status']}")


def _history_text(mem: dict, full: bool) -> str:
    turns = (mem.get("archive", []) + mem["turns"]) if full else mem["turns"]
    lines = [f"{'用户' if t['role'] == 'user' else '助手'}：{t['text']}" for t in turns]
    if not full and mem["summary"]:
        lines.insert(0, f"早前摘要：{mem['summary']}")
    return "\n".join(lines)


def _rules_text(rules: list[dict]) -> str:
    return "\n".join(f"[{r['id']}] {r['title']}：{r['content']}" for r in rules)


def build_context(order, rules, mem, profile) -> dict:
    cfg = S.get()["memory"]
    budget, share = cfg["token_budget"], cfg["budget_share"]
    k = S.get()["retrieval"]["k_inject"]
    raw = {"system": SYSTEM_PROMPT, "order": order_text(order), "rules": _rules_text(rules[:k]),
           "history": _history_text(mem, full=False), "profile": profile_text(profile)}
    full = {**raw, "rules": _rules_text(rules), "history": _history_text(mem, full=True),
            "profile": json.dumps(profile["data"], ensure_ascii=False) if profile else ""}
    blocks, used = {}, 0
    for name in PRIORITY:              # 高优先级先拿；没用完的额度顺延给后面的块
        remaining = budget - used
        quota = remaining if name == PRIORITY[-1] else min(remaining, max(int(budget * share[name]),
                                                                          tokens(raw[name]) if name in ("system", "order") else 0))
        text = raw[name]
        if tokens(text) > quota:
            text = text[:int(quota * 1.5)]
        t = tokens(text)
        blocks[name] = {"text": text, "tokens": t, "budget": quota, "truncated": text != raw[name]}
        used += t
    total = sum(b["tokens"] for b in blocks.values())
    full_total = sum(tokens(v) for v in full.values())
    return {"blocks": blocks, "total": total, "budget": budget, "full_total": full_total,
            "saved_pct": round(100 * (1 - total / full_total), 1) if full_total else 0.0}
```

- [ ] **Step 5：运行测试，确认通过**

Run: `uv run pytest tests/test_memory.py -v`
Expected: 7 passed

- [ ] **Step 6：提交**

```bash
git add air_agent/memory.py air_agent/data.py tests/test_memory.py
git commit -m "feat: add session memory, user profile and token budget"
```

### Task 8：方案生成与质量校验（工作五）

**Files:**
- Create: `air_agent/scheme.py`、`air_agent/verify.py`
- Test: `tests/test_scheme_verify.py`

`results` 里每个子诉求的结构（由 Task 10 的图节点产出）：
`{"intent", "text", "target_date", "status": {...}|None, "rules": [规则], "fee": 引擎结果|None, "error": str|None}`

- [ ] **Step 1：写失败测试 `tests/test_scheme_verify.py`**

```python
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
```

- [ ] **Step 2：运行测试，确认失败**

Run: `uv run pytest tests/test_scheme_verify.py -v`
Expected: FAIL，`ImportError: cannot import name 'scheme'`

- [ ] **Step 3：实现 `air_agent/scheme.py`**

```python
"""方案生成：模板（规则模式）或 LLM（DeepSeek 模式）。金额一律回填引擎结果。"""
from .state import FeeItem, SchemeOption, SchemeOutput

STATUS_LABEL = {"scheduled": "计划正常", "delayed": "延误", "cancelled": "已取消"}


def _option(r: dict) -> tuple[SchemeOption, str]:
    refs = [x["id"] for x in r["rules"][:1]]
    fee = r["fee"]
    if r.get("error"):
        return (SchemeOption(type=r["intent"], title="暂时无法办理", allowed=False, rule_refs=[], notes=r["error"]),
                f"{r['text']}：{r['error']}。")
    if r["intent"] == "STATUS":
        st = r["status"]
        delay = f"，延误{st['delay_min']}分钟" if st["status"] == "delayed" else ""
        return (SchemeOption(type="STATUS", title="航班状态", notes=STATUS_LABEL[st["status"]]),
                f"航班{st['flight_no']}（{st['date']} {st['dep_time']}）当前{STATUS_LABEL[st['status']]}{delay}。")
    if r["intent"] == "POLICY" or fee is None:
        top = r["rules"][0] if r["rules"] else None
        if not top:
            return SchemeOption(type="POLICY", title="未找到规则", allowed=False), "没有检索到相关规则。"
        return (SchemeOption(type="POLICY", title=top["title"], rule_refs=[top["id"]]),
                f"根据《{top['title']}》：{top['content']}")
    refs = [fee["rule_id"]] if any(x["id"] == fee["rule_id"] for x in r["rules"]) else refs
    if not fee["allowed"]:
        return (SchemeOption(type=r["intent"], title="不可办理", allowed=False, rule_refs=refs, notes=fee["reason"]),
                f"很抱歉，{fee['reason']}（依据规则{refs[0] if refs else ''}）。")
    if r["intent"] == "CHANGE":
        t = fee["target"]
        items = [FeeItem(name="改期费", amount=fee["change_fee"]), FeeItem(name="票面差价", amount=fee["fare_diff"])]
        head = "因航班非自愿变动，" if fee["involuntary"] else f"当前处于{fee['window_label']}档，"
        text = (f"{head}改签至{t['date']} {t['flight_no']}（{t['dep_time']}）：改期费{fee['change_fee']}元，"
                f"票面差价{fee['fare_diff']}元，合计需支付{fee['total']}元。")
        return (SchemeOption(type="CHANGE", title=f"改签至 {t['date']} {t['flight_no']}", fee_items=items,
                             total=fee["total"], rule_refs=refs), text)
    items = [FeeItem(name="退票费", amount=fee["refund_fee"]), FeeItem(name="机场建设费退还", amount=fee["tax"])]
    head = "因航班非自愿变动，" if fee["involuntary"] else f"当前处于{fee['window_label']}档，"
    text = (f"{head}退票：退票费{fee['refund_fee']}元，退还机场建设费{fee['tax']}元，"
            f"预计退款{fee['refund_amount']}元。")
    return (SchemeOption(type="REFUND", title="退票", fee_items=items, total=fee["refund_amount"], rule_refs=refs),
            text)


def generate(results: list[dict], feedback: list[str] | None = None, llm=None, context: dict | None = None) -> SchemeOutput:
    opts, texts = [], []
    for r in results:
        o, t = _option(r)
        opts.append(o)
        texts.append(t)
    out = SchemeOutput(options=opts, text="".join(texts))
    if llm is not None:
        polished = llm.write_scheme(out, results, feedback or [], context or {})
        if polished:
            out = enforce_amounts(polished, out)
    return out


def enforce_amounts(candidate: SchemeOutput, reference: SchemeOutput) -> SchemeOutput:
    """LLM 只负责措辞：结构化字段一律用模板（引擎）值覆盖。"""
    return SchemeOutput(options=reference.options, text=candidate.text)
```

- [ ] **Step 4：实现 `air_agent/verify.py`**

```python
"""工作五：金额后验、论断事实核查、引用合法性，计算置信度。"""
import re

AMOUNT = re.compile(r"(\d+(?:\.\d+)?)\s*元")
CLAIM_KEYS = re.compile(r"免费|免收|全额|不得|不可|可以|\d+次|\d+%|会员|积分|赔偿|补偿")


def _allowed_amounts(results):
    nums = set()
    for r in results:
        f = r.get("fee") or {}
        for k in ("change_fee", "fare_diff", "total", "refund_fee", "refund_amount", "tax", "fare"):
            if isinstance(f.get(k), (int, float)):
                nums.add(int(f[k]))
    return nums


def _rule_corpus(results):
    return "".join(f"{x['title']}{x['content']}" for r in results for x in r.get("rules", []))


def _supported(sentence: str, corpus: str) -> bool:
    for kw in CLAIM_KEYS.findall(sentence):
        if kw not in corpus:
            return False
    return True


def check(scheme: dict, results: list[dict], fact_checker=None) -> dict:
    errors = []
    allowed = _allowed_amounts(results)
    # 1. 文案金额必须来自引擎
    quoted = [int(float(x)) for x in AMOUNT.findall(scheme["text"])]
    bad_quotes = [q for q in quoted if q not in allowed]
    # 2. 结构化金额逐项比对
    for opt, r in zip(scheme["options"], results):
        fee = r.get("fee")
        if not fee or not fee.get("allowed") or opt["type"] not in ("CHANGE", "REFUND"):
            continue
        expect_total = fee["total"] if opt["type"] == "CHANGE" else fee["refund_amount"]
        if opt.get("total") != expect_total:
            errors.append(f"{opt['title']} 合计应为{expect_total}元，实际写了{opt.get('total')}元")
        names = {"改期费": fee.get("change_fee"), "票面差价": fee.get("fare_diff"), "退票费": fee.get("refund_fee")}
        for item in opt.get("fee_items", []):
            if item["name"] in names and item["amount"] != names[item["name"]]:
                errors.append(f"{item['name']}应为{names[item['name']]}元，实际写了{item['amount']}元")
    for q in bad_quotes:
        near = sorted(allowed, key=lambda a: abs(a - q))[:1]
        errors.append(f"文案金额{q}元没有工具依据" + (f"，应为{near[0]}元" if near else ""))
    quote_ok = not bad_quotes and not errors
    # 3. 论断事实核查
    corpus = _rule_corpus(results)
    sentences = [s for s in re.split(r"[。！？；]", scheme["text"]) if s.strip()]
    unsupported = [s for s in sentences if not _supported(s, corpus)]
    if fact_checker is not None:
        unsupported = fact_checker(sentences, corpus) or unsupported
    for s in unsupported:
        errors.append(f"删除无依据的论断：「{s.strip()}」")
    # 4. 引用必须在本次检索结果里
    retrieved = {x["id"] for r in results for x in r.get("rules", [])}
    bad_refs = [ref for o in scheme["options"] for ref in o.get("rule_refs", []) if ref not in retrieved]
    for ref in bad_refs:
        errors.append(f"引用的规则 {ref} 不在检索结果中")
    # 5. 需要依据的子诉求必须检索到规则
    no_rules = [r["text"] for r in results if r["intent"] != "STATUS" and not r.get("rules")]
    for t in no_rules:
        errors.append(f"「{t}」没有检索到依据规则，不能作答")
    support_ratio = 1 - len(unsupported) / len(sentences) if sentences else 1.0
    confidence = round(min(1.0 if quote_ok else 0.0, support_ratio, 0.0 if bad_refs or no_rules else 1.0), 2)
    return {"confidence": confidence, "quote_mismatch": not quote_ok, "bad_quotes": bad_quotes,
            "unsupported_claims": unsupported, "bad_refs": bad_refs, "errors": errors,
            "support_ratio": round(support_ratio, 2), "passed": confidence >= _threshold()}


def _threshold():
    from . import settings as S
    return S.get()["verify"]["threshold"]
```

- [ ] **Step 5：运行测试，确认通过**

Run: `uv run pytest tests/test_scheme_verify.py -v`
Expected: 6 passed。如果 `test_template_scheme_passes_verify` 因为 `unsupported_claims` 失败，检查模板文案里的关键词（如「免收」「全额」）是否出现在检索到的规则原文里；应当修模板措辞，不要放宽 `CLAIM_KEYS`。

- [ ] **Step 6：提交**

```bash
git add air_agent/scheme.py air_agent/verify.py tests/test_scheme_verify.py
git commit -m "feat: add scheme generation and verification"
```

### Task 9：DeepSeek 客户端与 ReAct 决策

**Files:**
- Create: `air_agent/llm.py`、`air_agent/planner.py`
- Test: `tests/test_planner.py`

- **规则模式**：按意图查固定计划表，选出第一个还没成功执行的工具。
- **DeepSeek 模式**：把工具 Schema 交给模型，由模型选择工具和参数。调用出错时这一步降级为规则模式（`source="fallback"`）。
- **`FakeLLM`**：按预设脚本回放 `react()` 的结果，用来测试 DeepSeek 模式的分支，不联网。

- [ ] **Step 1：写失败测试 `tests/test_planner.py`**

```python
import httpx
import pytest
from air_agent import data, llm, planner, query
from air_agent import settings as S

D = "20260926"


def _state(question, **kw):
    data.init()
    ctx = {"user_id": "u_zhang", "selected_order_id": f"ORD{D}001", "order_stack": [], "last_intent": None}
    p = query.understand(question, ctx)
    return {"parsed": p.model_dump(), "sub_idx": 0, "calls": [], "error_feedback": [], **kw}


def test_rule_plan_for_change():
    st = _state("改到大后天")
    seq = []
    for _ in range(4):
        a = planner.decide(st)
        seq.append(a["action"])
        st["calls"].append({"tool": a["action"], "args": a["args"], "ok": True, "output": {}})
    assert seq == ["tool_flight_status", "tool_rule_retrieve", "tool_recalc_fee", "finish"]


def test_rule_args_for_recalc():
    st = _state("改到大后天")
    st["calls"] = [{"tool": "tool_flight_status", "ok": True}, {"tool": "tool_rule_retrieve", "ok": True}]
    a = planner.decide(st)
    assert a["args"] == {"order_id": f"ORD{D}001", "action": "CHANGE", "target_date": "2026-09-29"}
    assert a["thought"]


def test_rule_retrieve_args_have_filters():
    st = _state("改到大后天")
    st["calls"] = [{"tool": "tool_flight_status", "ok": True}]
    a = planner.decide(st)
    assert a["args"]["airline"] == "MU" and a["args"]["fare_product"] == "Y_DISCOUNT"
    assert a["args"]["scenario"] == "VOLUNTARY_CHANGE"


def test_date_type_error_turns_into_clarify():
    st = _state("改到下个月初")
    st["error_feedback"] = [{"field": "target_date", "kind": "TYPE", "expected": "YYYY-MM-DD"}]
    a = planner.decide(st)
    assert a["action"] == "clarify" and a["args"]["field"] == "target_date"


def test_policy_plan():
    data.init()
    p = query.understand("东方云航特价票能退吗", {"user_id": "nobody", "selected_order_id": None,
                                               "order_stack": [], "last_intent": None})
    st = {"parsed": p.model_dump(), "sub_idx": 0, "calls": [], "error_feedback": []}
    assert planner.decide(st)["action"] == "tool_rule_retrieve"


def test_fake_llm_drives_decision():
    st = _state("改到大后天")
    fake = llm.FakeLLM(react=[{"thought": "先算钱", "action": "tool_recalc_fee",
                               "args": {"order_id": f"ORD{D}001", "action": "CHANGE", "target_date": "后天"}}])
    a = planner.decide(st, fake)
    assert a["source"] == "llm" and a["args"]["target_date"] == "后天"


def test_llm_error_falls_back_to_rules():
    st = _state("改到大后天")
    a = planner.decide(st, llm.FakeLLM(react=[llm.LLMError("timeout")]))
    assert a["source"] == "fallback" and a["action"] == "tool_flight_status"


def test_deepseek_client_parses_tool_call(monkeypatch):
    S.update({"mode": "deepseek", "llm": {"api_key": "sk-test-000000"}})

    def handler(request):
        return httpx.Response(200, json={"choices": [{"message": {"content": "需要先查规则", "tool_calls": [
            {"id": "c1", "type": "function", "function": {"name": "tool_rule_retrieve",
                                                         "arguments": "{\"query\": \"改签\"}"}}]}}],
            "usage": {"prompt_tokens": 100, "completion_tokens": 20}})

    client = llm.DeepSeekClient(transport=httpx.MockTransport(handler))
    a = client.react([{"role": "user", "content": "x"}])
    assert a == {"thought": "需要先查规则", "action": "tool_rule_retrieve", "args": {"query": "改签"}}
    assert client.usage == {"calls": 1, "prompt_tokens": 100, "completion_tokens": 20}


def test_get_client_requires_key():
    S.update({"mode": "deepseek"})
    with pytest.raises(llm.LLMNotConfigured):
        llm.get_client("deepseek")
    assert llm.get_client("offline") is None
```

- [ ] **Step 2：运行测试，确认失败**

Run: `uv run pytest tests/test_planner.py -v`
Expected: FAIL，`ImportError: cannot import name 'llm'`

- [ ] **Step 3：实现 `air_agent/llm.py`**

```python
"""DeepSeek（OpenAI 兼容）客户端与测试用 FakeLLM。任何异常都抛 LLMError，由调用方降级。"""
import json
import httpx
from pydantic import BaseModel, ValidationError
from . import settings as S
from .state import SchemeOutput, SubQuery


class LLMError(Exception):
    pass


class LLMNotConfigured(LLMError):
    pass


class _Intents(BaseModel):
    sub_queries: list[SubQuery]


class DeepSeekClient:
    def __init__(self, transport=None):
        cfg = S.get()["llm"]
        self.model = cfg["model"]
        self.http = httpx.Client(base_url=cfg["base_url"], timeout=cfg["timeout"], transport=transport,
                                 headers={"Authorization": f"Bearer {cfg['api_key']}"})
        self.usage = {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0}

    def chat(self, messages, tools=None, json_mode=False) -> dict:
        body = {"model": self.model, "messages": messages, "temperature": 0}
        if tools:
            body["tools"] = tools
        if json_mode:
            body["response_format"] = {"type": "json_object"}
        try:
            r = self.http.post("/chat/completions", json=body)
            r.raise_for_status()
            payload = r.json()
        except (httpx.HTTPError, ValueError) as e:
            raise LLMError(f"{type(e).__name__}: {e}") from e
        u = payload.get("usage", {})
        self.usage["calls"] += 1
        self.usage["prompt_tokens"] += u.get("prompt_tokens", 0)
        self.usage["completion_tokens"] += u.get("completion_tokens", 0)
        return payload["choices"][0]["message"]

    def _json(self, system, user) -> dict:
        msg = self.chat([{"role": "system", "content": system}, {"role": "user", "content": user}], json_mode=True)
        try:
            return json.loads(msg.get("content") or "{}")
        except json.JSONDecodeError as e:
            raise LLMError(f"JSON 解析失败：{e}") from e

    def react(self, messages, tools=None) -> dict:
        from .tools import llm_specs
        msg = self.chat(messages, tools=tools or llm_specs())
        calls = msg.get("tool_calls") or []
        if not calls:
            return {"thought": msg.get("content") or "", "action": "finish", "args": {}}
        fn = calls[0]["function"]
        try:
            args = json.loads(fn.get("arguments") or "{}")
        except json.JSONDecodeError:
            args = {"_raw": fn.get("arguments")}
        return {"thought": msg.get("content") or "", "action": fn["name"], "args": args}

    def parse_intents(self, question, ctx, rule_subs):
        try:
            out = self._json(
                "把乘客的退改签诉求拆成子诉求，输出 JSON：{\"sub_queries\":[{\"intent\":\"CHANGE|REFUND|POLICY|STATUS\","
                "\"text\":\"子诉求原文\",\"target_date\":\"改签目标日期原文或null\"}]}。不要编造日期。",
                f"上一轮意图：{ctx.get('last_intent')}\n乘客：{question}")
            subs = _Intents.model_validate(out).sub_queries
        except (LLMError, ValidationError):
            return None
        from .query import resolve_date
        for s in subs:
            if s.target_date:
                s.target_date = resolve_date(s.target_date) or s.target_date
        return subs

    def summarize(self, turns):
        try:
            return self._json("把对话压缩成一句话摘要，保留订单、日期、金额。输出 {\"summary\": \"...\"}",
                              json.dumps(turns, ensure_ascii=False)).get("summary")
        except LLMError:
            return None

    def write_scheme(self, draft: SchemeOutput, results, feedback, context):
        prompt = (f"上下文：\n{json.dumps({k: v['text'] for k, v in context.get('blocks', {}).items()}, ensure_ascii=False)}\n"
                  f"方案草稿：{draft.model_dump_json()}\n"
                  f"上次校验的问题：{feedback}\n"
                  "请把 text 改写得更自然，金额必须与草稿逐字一致，不得新增规则里没有的承诺。输出 {\"text\": \"...\"}")
        try:
            out = self._json("你是航空公司退改签客服，只能依据给定规则和金额作答。", prompt)
            return SchemeOutput(options=draft.options, text=out["text"])
        except (LLMError, KeyError, ValidationError):
            return None

    def fact_check(self, sentences, corpus):
        try:
            out = self._json("判断每句话能否在规则原文中找到依据。输出 {\"unsupported\": [\"无依据的句子\"]}",
                             f"规则原文：{corpus}\n句子：{json.dumps(sentences, ensure_ascii=False)}")
            return [s for s in out.get("unsupported", []) if s in sentences]
        except LLMError:
            return None


class FakeLLM:
    """按脚本回放 react()；其余方法返回 None，让调用方走规则逻辑。"""

    def __init__(self, react=None, scheme_texts=None):
        self.script = list(react or [])
        self.scheme_texts = list(scheme_texts or [])   # 每次 write_scheme 取一个；None 表示用模板
        self.usage = {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0}

    def react(self, messages, tools=None):
        self.usage["calls"] += 1
        if not self.script:
            raise LLMError("脚本已用完")
        item = self.script.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    def parse_intents(self, *a):
        return None

    def summarize(self, *a):
        return None

    def write_scheme(self, draft, *a):
        text = self.scheme_texts.pop(0) if self.scheme_texts else None
        return SchemeOutput(options=draft.options, text=text) if text else None

    def fact_check(self, *a):
        return None


def get_client(mode: str):
    if mode != "deepseek":
        return None
    if not S.get()["llm"]["api_key"]:
        raise LLMNotConfigured("DeepSeek 模式需要先在配置中心填写 API Key")
    return DeepSeekClient()
```

- [ ] **Step 4：实现 `air_agent/planner.py`**

```python
"""ReAct 决策：输出 {thought, action, args, source}。action 为工具名、finish、clarify 或 human。"""
import json
from . import data, query, seed
from .llm import LLMError
from .state import INTENT_LABEL, ParsedQuery, SubQuery

PLANS = {"CHANGE": ["tool_flight_status", "tool_rule_retrieve", "tool_recalc_fee"],
         "REFUND": ["tool_flight_status", "tool_rule_retrieve", "tool_recalc_fee"],
         "POLICY": ["tool_rule_retrieve"],
         "STATUS": ["tool_flight_status"]}
THOUGHTS = {"tool_flight_status": "先确认航班状态：取消或延误 2 小时以上属于非自愿退改，费用规则不同。",
            "tool_rule_retrieve": "检索适用的退改规则，确定收费档位和依据。",
            "tool_recalc_fee": "规则已确认，调用运价引擎计算金额（金额只能来自引擎）。",
            "finish": "所需信息已齐全，进入方案生成。"}


def current(state) -> tuple[ParsedQuery, SubQuery]:
    p = ParsedQuery.model_validate(state["parsed"])
    return p, p.sub_queries[state.get("sub_idx", 0)]


def build_args(tool, p: ParsedQuery, sub: SubQuery) -> dict:
    if tool == "tool_flight_status":
        o = data.get_order(p.order_id)
        return {"flight_no": o["flight_no"], "date": o["date"]}
    if tool == "tool_rule_retrieve":
        scen = query.scenario_for(p, sub)
        name = seed.AIRLINES.get(p.airline or "", "")
        prod = seed.PRODUCTS.get(p.fare_product or "", ("", ""))[1]
        args = {"query": f"{name}{prod}{query.KIND_LABEL.get(scen or '', '')} {sub.text}".strip()}
        for k in ("airline", "fare_product"):
            if getattr(p, k):
                args[k] = getattr(p, k)
        if scen:
            args["scenario"] = scen
        return args
    if tool == "tool_recalc_fee":
        args = {"order_id": p.order_id, "action": sub.intent}
        if sub.intent == "CHANGE":
            args["target_date"] = sub.target_date
        return args
    return {}


def rule_decide(state) -> dict:
    p, sub = current(state)
    for err in state.get("error_feedback") or []:
        if err.get("kind") == "SKIP_RULE":      # 防跳步打回：按计划表继续即可
            continue
        if err.get("field") == "target_date":
            return {"thought": f"日期「{sub.target_date}」无法换算成具体日期，需要向乘客确认。",
                    "action": "clarify", "args": {"field": "target_date"}, "source": "rule"}
        return {"thought": f"工具参数错误且无法自动修正：{err.get('message', err)}", "action": "human",
                "args": {}, "source": "rule"}
    done = {c["tool"] for c in state.get("calls", []) if c.get("ok")}
    for tool in PLANS[sub.intent]:
        if tool not in done:
            return {"thought": THOUGHTS[tool], "action": tool, "args": build_args(tool, p, sub), "source": "rule"}
    return {"thought": THOUGHTS["finish"], "action": "finish", "args": {}, "source": "rule"}


def llm_messages(state) -> list[dict]:
    p, sub = current(state)
    ctx = state.get("context", {}).get("blocks", {})
    sys = ctx.get("system", {}).get("text", "")
    obs = [{"tool": c["tool"], "args": c.get("args"), "ok": c.get("ok"),
            "result": c.get("output") if c.get("ok") else c.get("errors") or c.get("error")}
           for c in state.get("calls", [])]
    user = (f"{ctx.get('order', {}).get('text', '')}\n用户画像：{ctx.get('profile', {}).get('text', '')}\n"
            f"会话：{ctx.get('history', {}).get('text', '')}\n"
            f"当前子诉求（{INTENT_LABEL[sub.intent]}）：{sub.text}；目标日期：{sub.target_date}\n"
            f"已执行：{json.dumps(obs, ensure_ascii=False)}\n"
            f"上次参数错误：{json.dumps(state.get('error_feedback') or [], ensure_ascii=False)}\n"
            f"今天是 {query.today().isoformat()}。先说一句思考，再调用一个工具；信息齐全时调用 finish。"
            f"算钱前必须先检索规则；日期必须是 YYYY-MM-DD。")
    return [{"role": "system", "content": sys}, {"role": "user", "content": user}]


def decide(state, llm=None) -> dict:
    if llm is None:
        return rule_decide(state)
    try:
        a = llm.react(llm_messages(state))
        return {**a, "source": "llm"}
    except LLMError as e:
        a = rule_decide(state)
        return {**a, "source": "fallback", "fallback_reason": str(e)}
```

- [ ] **Step 5：运行测试，确认通过**

Run: `uv run pytest tests/test_planner.py -v`
Expected: 9 passed

- [ ] **Step 6：提交**

```bash
git add air_agent/llm.py air_agent/planner.py tests/test_planner.py
git commit -m "feat: add DeepSeek client, FakeLLM and ReAct planner"
```

### Task 10：出票工具（幂等）

**Files:**
- Modify: `air_agent/data.py`（SCHEMA 加 `tickets` 表）、`air_agent/tools.py`（加 `issue_ticket`）
- Test: `tests/test_issue_ticket.py`

- [ ] **Step 1：写失败测试 `tests/test_issue_ticket.py`**

```python
from air_agent import data, engine, tools

D = "20260926"


def test_change_ticket_updates_order_and_is_idempotent():
    data.init()
    before = data.get_order(f"ORD{D}001")
    fee = engine.calc(before["order_id"], "CHANGE", target_date="2026-09-29")
    args = {"order_id": before["order_id"], "action": "CHANGE", "confirm_id": "confirm-001",
            "target_date": "2026-09-29"}
    first = tools.issue_ticket(args)
    second = tools.issue_ticket(args)
    after = data.get_order(before["order_id"])
    assert first["ticket_no"] == second["ticket_no"] and second["idempotent"] is True
    assert after["date"] == "2026-09-29" and after["ticket_no"] == first["ticket_no"]
    assert first["amount"] == fee["total"]


def test_refund_ticket():
    data.init()
    o = data.get_order(f"ORD{D}003")
    r = tools.issue_ticket({"order_id": o["order_id"], "action": "REFUND", "confirm_id": "confirm-002"})
    assert r["status"] == "REFUNDED" and data.get_order(o["order_id"])["status"] == "REFUNDED"


def test_not_allowed_raises():
    data.init()
    gp = next(o for o in data.list_orders() if o["fare_product"] == "GROUP")
    r = tools.issue_ticket({"order_id": gp["order_id"], "action": "REFUND", "confirm_id": "confirm-003"})
    assert r["status"] == "REJECTED"
```

- [ ] **Step 2：运行测试，确认失败**

Run: `uv run pytest tests/test_issue_ticket.py -v`
Expected: FAIL，`AttributeError: module 'air_agent.tools' has no attribute 'issue_ticket'`

- [ ] **Step 3：在 `air_agent/data.py` 的 `SCHEMA` 末尾加表**

```sql
CREATE TABLE IF NOT EXISTS tickets(confirm_id TEXT PRIMARY KEY, order_id TEXT, action TEXT, result TEXT, created_at TEXT);
```

- [ ] **Step 4：在 `air_agent/tools.py` 末尾加 `issue_ticket`**

```python
def issue_ticket(args: dict) -> dict:
    """出票/退票。金额在出票时由引擎重算；同一 confirm_id 只办理一次。"""
    import json
    from . import settings as S
    model, errs = validate("tool_issue_ticket", args)
    if errs:
        return {"status": "REJECTED", "reason": errs[0]["message"]}
    a = model.model_dump()
    with data.connect() as c:
        row = c.execute("SELECT result FROM tickets WHERE confirm_id=?", (a["confirm_id"],)).fetchone()
    if row:
        return {**json.loads(row["result"]), "idempotent": True}
    try:
        fee = engine.calc(a["order_id"], a["action"], a["target_date"], a["target_flight"])
    except engine.EngineError as e:
        return {"status": "REJECTED", "reason": str(e)}
    if not fee["allowed"]:
        return {"status": "REJECTED", "reason": fee["reason"]}
    order = data.get_order(a["order_id"])
    if a["action"] == "CHANGE":
        t = fee["target"]
        ticket_no = f"781-{abs(hash((a['confirm_id'], t['flight_no']))) % 10**10:010d}"
        data.update_order(a["order_id"], flight_no=t["flight_no"], date=t["date"],
                          fare=max(order["fare"], t["fare"]), ticket_no=ticket_no)
        result = {"status": "CHANGED", "action": "CHANGE", "order_id": a["order_id"], "ticket_no": ticket_no,
                  "amount": fee["total"], "target": t}
    else:
        data.update_order(a["order_id"], status="REFUNDED")
        result = {"status": "REFUNDED", "action": "REFUND", "order_id": a["order_id"],
                  "ticket_no": order["ticket_no"], "amount": fee["refund_amount"]}
    with data.connect() as c:
        c.execute("INSERT INTO tickets VALUES(?,?,?,?,?)", (a["confirm_id"], a["order_id"], a["action"],
                                                           json.dumps(result, ensure_ascii=False), S.now().isoformat()))
    return {**result, "idempotent": False}
```

注意：Python 的 `hash()` 对字符串有随机盐，跨进程结果会变；这里只在第一次出票时生成票号并存进 `tickets` 表，之后都读表，所以不影响幂等。

- [ ] **Step 5：运行测试，确认通过**

Run: `uv run pytest tests/test_issue_ticket.py -v`
Expected: 3 passed

- [ ] **Step 6：提交**

```bash
git add air_agent/data.py air_agent/tools.py tests/test_issue_ticket.py
git commit -m "feat: add idempotent ticket issuing"
```

### Task 11：LangGraph 状态图（工作一，串起全部模块）

**Files:**
- Modify: `air_agent/state.py`（trace 改用「按轮重置」的 reducer，补充字段）
- Create: `air_agent/graph.py`
- Test: `tests/test_graph.py`、`tests/test_graph_llm.py`

**节点**（与 work-items.json 对齐，另加 `sub_done`、`step_back`、`await_confirm`、`tool_issue_ticket`）：
`user_input → memory_node → order_align → query_understand →（clarify）→ agent_node ⇄ schema_validate → checkpoint_gate → tool_exec →（step_back）→ … → sub_done → tool_scheme_gen → verify_node →（retry_node）→ finish | human_review → finish →（await_confirm → tool_issue_ticket）`

**计数器**：
- `tool_retries`：工具参数校验连续失败的次数，工具执行成功后清零，超过 `max_tool_retries` 就转人工；
- `gate_blocks`：被防跳步打回的次数，超过 2 次转人工；
- `rounds`：当前子诉求的 ReAct 轮数，超过 `max_rounds` 转人工；
- `retries`：方案生成被校验打回后的重试次数；
- `clarify_count`：追问次数，达到 3 次转人工。

- [ ] **Step 1：修改 `air_agent/state.py`**

把文件顶部的 `import operator` 换成下面的 reducer，并把 `AgentState` 的 `trace` 一行改掉，同时补充字段：

```python
def trace_reducer(old: list | None, new: list | None) -> list:
    """每轮从 user_input 开始重置 trace；resume 时继续追加。"""
    new = new or []
    if new and new[0].get("node") == "user_input":
        return list(new)
    return (old or []) + new
```

```python
    # 在 AgentState 中替换/新增：
    memory: dict
    clarified: dict               # 追问得到的值：{order_id | target_date | intent: value}
    clarify_req: dict | None      # agent_node 发起的追问
    clarify_count: int
    gate_blocks: int
    stepped_back: bool
    confirm: dict | None
    trace: Annotated[list[dict], trace_reducer]
```

- [ ] **Step 2：写失败测试 `tests/test_graph.py`（规则模式）**

```python
import pytest
from air_agent import data, engine, graph

D = "20260926"
O1, O3 = f"ORD{D}001", f"ORD{D}003"


@pytest.fixture(autouse=True)
def _fresh():
    data.init()
    graph.reset_graph()
    yield
    graph.reset_graph()


def nodes(res):
    return [e["node"] for e in res["trace"]]


def test_single_change_full_path():
    res = graph.run_turn("s1", "u_zhang", "帮我改到大后天", order_id=O1)
    assert res["outcome"] == "ANSWER"
    assert res["interrupt"]["kind"] == "confirm"
    expect = engine.calc(O1, "CHANGE", target_date="2026-09-29")["total"]
    assert res["final"]["options"][0]["total"] == expect
    ns = nodes(res)
    for n in ["user_input", "memory_node", "order_align", "query_understand", "agent_node", "schema_validate",
              "checkpoint_gate", "tool_exec", "sub_done", "tool_scheme_gen", "verify_node", "finish"]:
        assert n in ns
    tools_called = [e["output"]["action"] for e in res["trace"] if e["node"] == "agent_node"]
    assert tools_called == ["tool_flight_status", "tool_rule_retrieve", "tool_recalc_fee", "finish"]


def test_multi_intent_two_options():
    res = graph.run_turn("s2", "u_zhang", "改到大后天，顺便看看退票多少钱", order_id=O1)
    assert [o["type"] for o in res["final"]["options"]] == ["CHANGE", "REFUND"]
    assert nodes(res).count("sub_done") == 2


def test_clarify_order_then_resume():
    res = graph.run_turn("s3", "u_zhang", "我要退票")
    assert res["interrupt"]["kind"] == "clarify" and res["interrupt"]["field"] == "order_id"
    res = graph.resume("s3", {"value": O3})
    assert res["outcome"] == "ANSWER" and res["final"]["options"][0]["type"] == "REFUND"


def test_unparseable_date_schema_error_then_clarify():
    res = graph.run_turn("s4", "u_zhang", "改到下个月初", order_id=O1)
    errs = [e for e in res["trace"] if e["node"] == "schema_validate" and e["status"] == "error"]
    assert errs and "TYPE" in errs[0]["note"]
    assert res["interrupt"]["kind"] == "clarify" and res["interrupt"]["field"] == "target_date"
    res = graph.resume("s4", {"value": "2026-10-02"})
    assert res["outcome"] == "ANSWER"
    assert res["results"][0]["target_date"] == "2026-10-02"


def test_confirm_issues_ticket():
    graph.run_turn("s5", "u_zhang", "帮我改到大后天", order_id=O1)
    res = graph.resume("s5", {"confirm": True, "index": 0, "confirm_id": "confirm-s5"})
    assert res["outcome"] == "TICKETED"
    assert data.get_order(O1)["date"] == "2026-09-29"
    assert res["interrupt"] is None


def test_cancel_confirm():
    graph.run_turn("s6", "u_zhang", "帮我改到大后天", order_id=O1)
    res = graph.resume("s6", {"confirm": False})
    assert res["outcome"] == "CANCELLED" and data.get_order(O1)["date"] != "2026-09-29"


def test_follow_up_uses_memory():
    graph.run_turn("s7", "u_zhang", "帮我改到大后天", order_id=O1)
    res = graph.run_turn("s7", "u_zhang", "那退票呢")
    assert res["parsed"]["order_id"] == O1
    assert res["final"]["options"][0]["type"] == "REFUND"
    assert res["memory"]["order_stack"][0] == O1


def test_off_topic_goes_human():
    res = graph.run_turn("s8", "u_wang", "宠物能带上飞机吗")
    assert res["outcome"] == "HUMAN"
    assert any("依据规则" in r for r in res["final"]["reasons"])
    assert "retry_node" in nodes(res)


def test_cancelled_flight_is_free_in_answer():
    o = next(x for x in data.list_orders("u_wang")
             if data.get_flight(x["flight_no"], x["date"])["status"] == "cancelled")
    res = graph.run_turn("s9", "u_wang", "航班取消了，帮我退票", order_id=o["order_id"])
    opt = res["final"]["options"][0]
    assert opt["total"] == o["fare"] + o["tax"] and "GEN-CANCEL" in opt["rule_refs"]


def test_resume_after_restart():
    graph.run_turn("s10", "u_zhang", "我要退票")
    graph.reset_graph()                 # 模拟服务重启：新连接，从 SQLite 检查点恢复
    assert graph.pending_interrupt("s10")["kind"] == "clarify"
    res = graph.resume("s10", {"value": O3})
    assert res["outcome"] == "ANSWER"
```

- [ ] **Step 3：写失败测试 `tests/test_graph_llm.py`（DeepSeek 模式，用 FakeLLM）**

```python
import pytest
from air_agent import data, graph, llm

D = "20260926"
O1 = f"ORD{D}001"
FEE = lambda date: {"order_id": O1, "action": "CHANGE", "target_date": date}


@pytest.fixture(autouse=True)
def _fresh():
    data.init()
    graph.reset_graph()
    yield
    graph.reset_graph()


def act(tool, args=None, thought="..."):
    return {"thought": thought, "action": tool, "args": args or {}}


def test_schema_error_feedback_then_success():
    fake = llm.FakeLLM(react=[
        act("tool_rule_retrieve", {"query": "折扣票改签", "airline": "MU", "fare_product": "Y_DISCOUNT"}),
        act("tool_recalc_fee", FEE("后天")),                  # 相对日期 → TYPE 错误
        act("tool_recalc_fee", FEE("2026-09-28")),            # 带着反馈修正
        act("finish"),
    ])
    res = graph.run_turn("l1", "u_zhang", "改到后天", order_id=O1, mode="deepseek", client=fake)
    assert res["outcome"] == "ANSWER"
    sv = [e["status"] for e in res["trace"] if e["node"] == "schema_validate"]
    assert sv == ["ok", "error", "ok"]


def test_gate_blocks_fee_before_rules():
    fake = llm.FakeLLM(react=[
        act("tool_recalc_fee", FEE("2026-09-28")),            # 跳步：没查规则就算钱
        act("tool_rule_retrieve", {"query": "折扣票改签", "airline": "MU"}),
        act("tool_recalc_fee", FEE("2026-09-28")),
        act("finish"),
    ])
    res = graph.run_turn("l2", "u_zhang", "改到后天", order_id=O1, mode="deepseek", client=fake)
    gates = [e["status"] for e in res["trace"] if e["node"] == "checkpoint_gate"]
    assert gates[0] == "blocked" and res["outcome"] == "ANSWER"


def test_wrong_amount_text_triggers_retry_then_passes():
    steps = [act("tool_rule_retrieve", {"query": "折扣票改签", "airline": "MU", "fare_product": "Y_DISCOUNT"}),
             act("tool_recalc_fee", FEE("2026-09-28")), act("finish")]
    fake = llm.FakeLLM(react=steps, scheme_texts=["改签只需支付1元。", None])
    res = graph.run_turn("l3", "u_zhang", "改到后天", order_id=O1, mode="deepseek", client=fake)
    ns = [e["node"] for e in res["trace"]]
    assert "retry_node" in ns and res["outcome"] == "ANSWER"
    assert res["final"]["text"] != "改签只需支付1元。"


def test_llm_failure_falls_back_to_rules():
    fake = llm.FakeLLM(react=[llm.LLMError("timeout")] * 10)
    res = graph.run_turn("l4", "u_zhang", "改到后天", order_id=O1, mode="deepseek", client=fake)
    assert res["outcome"] == "ANSWER"
    assert all(e["output"]["source"] == "fallback" for e in res["trace"] if e["node"] == "agent_node")


def test_round_limit_goes_human():
    fake = llm.FakeLLM(react=[act("tool_flight_status", {"flight_no": "MU5101", "date": "2026-09-28"})] * 10)
    res = graph.run_turn("l5", "u_zhang", "改到后天", order_id=O1, mode="deepseek", client=fake)
    assert res["outcome"] == "HUMAN"
```

- [ ] **Step 4：运行测试，确认失败**

Run: `uv run pytest tests/test_graph.py tests/test_graph_llm.py -v`
Expected: FAIL，`ImportError: cannot import name 'graph'`

- [ ] **Step 5：实现 `air_agent/graph.py`**

```python
"""LangGraph 状态图：工作一的编排，串起 Query 理解、检索、记忆、生成与校验。

阅读顺序：build_graph() 看节点和边 → 各节点函数 → run_turn()/resume() 看如何驱动与流式输出。
interrupt 注意：节点在 resume 后会从头重新执行，所以 interrupt() 之前不能有副作用。
"""
import sqlite3
import time
import uuid
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt
from . import data, llm as llm_mod, memory, planner, query, retrieval, tools
from . import scheme as scheme_mod
from . import settings as S
from . import verify as verify_mod
from .state import AgentState, ParsedQuery

REACT_TOOLS = {"tool_flight_status", "tool_rule_retrieve", "tool_recalc_fee"}
_clients: dict = {}          # session_id -> LLM 客户端（不写进检查点）
_graph = None


def _llm(state):
    return _clients.get(state["session_id"])


def ev(node, t0, inp=None, out=None, note="", status="ok"):
    return {"node": node, "status": status, "ms": round((time.perf_counter() - t0) * 1000, 1),
            "at": S.now().isoformat(), "input": inp, "output": out, "note": note}


def _qctx(state):
    mem, cl = state.get("memory") or {}, state.get("clarified") or {}
    return {"user_id": state["user_id"], "selected_order_id": cl.get("order_id") or state.get("order_id"),
            "order_stack": mem.get("order_stack", []), "last_intent": mem.get("last_intent"),
            "forced_date": cl.get("target_date"), "forced_intent": cl.get("intent")}


# ---------------- 节点 ----------------
def user_input(state):
    t0 = time.perf_counter()
    return {"sub_idx": 0, "calls": [], "results": [], "rounds": 0, "tool_retries": 0, "gate_blocks": 0,
            "retries": 0, "feedback": [], "error_feedback": [], "clarified": {}, "clarify_req": None,
            "clarify_count": 0, "stepped_back": False, "outcome": "", "final": {}, "scheme": {}, "verify": {},
            "confirm": None,
            "trace": [ev("user_input", t0, {"question": state["question"], "order_id": state.get("order_id"),
                                            "mode": state.get("mode")}, None, "接收乘客诉求")]}


def memory_node(state):
    t0 = time.perf_counter()
    mem = memory.load_session(state["session_id"])
    view = {"order_stack": mem["order_stack"], "summary": mem["summary"], "turns": len(mem["turns"]),
            "last_intent": mem["last_intent"], "boost": memory.profile_boost(state["user_id"])}
    return {"memory": mem, "trace": [ev("memory_node", t0, None, view, "加载短期记忆与用户画像")]}


def order_align(state):
    t0 = time.perf_counter()
    oid = query.resolve_order(state["question"], _qctx(state))
    out = None
    if oid:
        o = data.get_order(oid)
        out = {"order_id": oid, "airline": o["airline"], "cabin": o["cabin"], "fare_product": o["fare_product"],
               "label": query.order_label(o)}
    return {"trace": [ev("order_align", t0, {"selected": state.get("order_id")}, out,
                         "订单字段对齐" if oid else "未能确定订单", "ok" if oid else "warn")]}


def query_understand(state):
    t0 = time.perf_counter()
    p = query.understand(state["question"], _qctx(state), _llm(state))
    order = data.get_order(p.order_id) if p.order_id else None
    ctx = memory.build_context(order, [], state["memory"], data.get_profile(state["user_id"]))
    note = f"{len(p.sub_queries)} 个子诉求，置信度 {p.confidence}" + ("，需要追问" if p.clarify else "")
    return {"parsed": p.model_dump(), "context": ctx,
            "trace": [ev("query_understand", t0, {"question": state["question"]}, p.model_dump(), note)]}


def clarify(state):
    t0 = time.perf_counter()
    req = state.get("clarify_req") or ParsedQuery.model_validate(state["parsed"]).clarify
    if state.get("clarify_count", 0) >= 3:
        return {"verify": {"errors": ["多次追问仍无法确定诉求"]}, "outcome": "HUMAN",
                "trace": [ev("clarify", t0, req, None, "追问次数已达上限", "error")]}
    answer = interrupt({"kind": "clarify", **req})
    value = answer.get("value") if isinstance(answer, dict) else answer
    return {"clarified": {**(state.get("clarified") or {}), req["field"]: value}, "clarify_req": None,
            "clarify_count": state.get("clarify_count", 0) + 1, "calls": [], "results": [], "sub_idx": 0,
            "rounds": 0, "tool_retries": 0, "gate_blocks": 0, "error_feedback": [], "stepped_back": False,
            "trace": [ev("clarify", t0, req, {"field": req["field"], "value": value}, "已获得乘客补充信息")]}


def agent_node(state):
    t0 = time.perf_counter()
    _, sub = planner.current(state)
    rounds = state.get("rounds", 0) + 1
    if rounds > S.get()["planner"]["max_rounds"]:
        a = {"thought": "已达到 ReAct 轮数上限，转人工", "action": "human", "args": {}, "source": "guard"}
    else:
        a = planner.decide(state, _llm(state))
    upd = {"rounds": rounds, "next_action": a,
           "trace": [ev("agent_node", t0, {"sub_query": sub.text, "intent": sub.intent, "round": rounds}, a,
                        f"第 {rounds} 轮 → {a['action']}")]}
    if a["action"] == "clarify":
        upd["clarify_req"] = {"field": "target_date", "options": query.date_options(),
                              "question": f"「{sub.target_date}」无法确定具体日期，想改签到哪一天？"}
    if a["action"] == "human":
        upd["verify"] = {"errors": [a["thought"]]}
    return upd


def schema_validate(state):
    t0 = time.perf_counter()
    a = state["next_action"]
    model, errs = tools.validate(a["action"], a["args"])
    if errs:
        call = {"tool": a["action"], "args": a["args"], "ok": False, "kind": errs[0]["kind"], "errors": errs}
        return {"tool_retries": state.get("tool_retries", 0) + 1, "error_feedback": errs,
                "calls": state["calls"] + [call],
                "trace": [ev("schema_validate", t0, {"tool": a["action"], "args": a["args"]}, errs,
                             f"{errs[0]['kind']}：{errs[0]['message']}", "error")]}
    clean = model.model_dump(exclude_none=True)
    return {"error_feedback": [], "next_action": {**a, "args": clean},
            "trace": [ev("schema_validate", t0, {"tool": a["action"], "args": a["args"]}, clean, "参数校验通过")]}


def checkpoint_gate(state):
    t0 = time.perf_counter()
    a = state["next_action"]
    done = {c["tool"] for c in state["calls"] if c.get("ok")}
    if a["action"] == "tool_recalc_fee" and "tool_rule_retrieve" not in done:
        fb = [{"field": "plan", "kind": "SKIP_RULE", "message": "计算金额前必须先调用 tool_rule_retrieve 检索规则"}]
        return {"gate_blocks": state.get("gate_blocks", 0) + 1, "error_feedback": fb, "gate_reason": fb[0]["message"],
                "trace": [ev("checkpoint_gate", t0, {"tool": a["action"], "done": sorted(done)}, None,
                             fb[0]["message"], "blocked")]}
    return {"gate_reason": "",
            "trace": [ev("checkpoint_gate", t0, {"tool": a["action"], "done": sorted(done)}, None, "必经节点检查通过")]}


def tool_exec(state):
    t0 = time.perf_counter()
    a = state["next_action"]
    r = tools.run_tool(a["action"], a["args"], memory.profile_boost(state["user_id"]))
    call = {"tool": a["action"], "args": a["args"], **r}
    upd = {"calls": state["calls"] + [call], "error_feedback": []}
    if r["ok"]:
        upd["tool_retries"] = 0
        note = {"tool_rule_retrieve": lambda o: f"命中 {len(o['rules'])} 条规则（{o['backend']}）",
                "tool_recalc_fee": lambda o: f"引擎计算完成：{o['rule_id']}",
                "tool_flight_status": lambda o: f"航班状态 {o['status']}"}[a["action"]](r["output"])
        upd["trace"] = [ev("tool_exec", t0, {"tool": a["action"], "args": a["args"]}, r["output"], note)]
    else:
        upd["trace"] = [ev("tool_exec", t0, {"tool": a["action"], "args": a["args"]}, r,
                           r.get("error", {}).get("message", "工具执行失败"), "error")]
    return upd


def step_back(state):
    t0 = time.perf_counter()
    p, sub = planner.current(state)
    last = state["calls"][-1]
    filters = {k: last["args"][k] for k in ("airline", "cabin", "fare_product", "scenario") if last["args"].get(k)}
    boost = memory.profile_boost(state["user_id"])
    q = query.step_back_query(p, sub)
    res, used = retrieval.search(q, filters, boost), "step_back"
    if not res["rules"]:
        q = query.hyde_text(p, sub)
        res, used = retrieval.search(q, filters, boost), "hyde"
    old = last["output"]["rules"]
    merged = old + [x for x in res["rules"] if x["id"] not in {o["id"] for o in old}]
    merged = merged[:S.get()["retrieval"]["k_inject"]]
    calls = state["calls"][:-1] + [{**last, "output": {**last["output"], "rules": merged},
                                    "rewrite": {"type": used, "query": q}}]
    return {"calls": calls, "stepped_back": True,
            "trace": [ev("step_back", t0, {"query": q, "type": used}, [x["id"] for x in merged],
                         f"{'Step-back 泛化' if used == 'step_back' else 'HyDE 扩写'}后共 {len(merged)} 条")]}


def sub_done(state):
    t0 = time.perf_counter()
    _, sub = planner.current(state)
    get = lambda tool: next((c for c in reversed(state["calls"]) if c["tool"] == tool and c.get("ok")), None)
    err = next((c for c in state["calls"] if c.get("kind") == "ENGINE"), None)
    rr, fs, fee = get("tool_rule_retrieve"), get("tool_flight_status"), get("tool_recalc_fee")
    result = {"intent": sub.intent, "text": sub.text, "target_date": sub.target_date,
              "status": fs["output"] if fs else None,
              "rules": rr["output"]["rules"] if rr else [], "funnel": rr["output"]["funnel"] if rr else [],
              "retrieval": {k: rr["output"][k] for k in ("backend", "degraded")} if rr else None,
              "rewrite": rr.get("rewrite") if rr else None,
              "fee": fee["output"] if fee else None, "error": err["error"]["message"] if err else None,
              "calls": [{k: c.get(k) for k in ("tool", "args", "ok", "kind")} for c in state["calls"]]}
    return {"results": state["results"] + [result], "sub_idx": state["sub_idx"] + 1, "calls": [], "rounds": 0,
            "tool_retries": 0, "gate_blocks": 0, "error_feedback": [], "stepped_back": False,
            "trace": [ev("sub_done", t0, {"sub_query": sub.text}, {"rules": [x["id"] for x in result["rules"]],
                                                                  "fee": result["fee"], "error": result["error"]},
                         f"子诉求完成：{sub.intent}")]}


def tool_scheme_gen(state):
    t0 = time.perf_counter()
    p = ParsedQuery.model_validate(state["parsed"])
    order = data.get_order(p.order_id) if p.order_id else None
    rules = [x for r in state["results"] for x in r["rules"]]
    ctx = memory.build_context(order, rules, state["memory"], data.get_profile(state["user_id"]))
    out = scheme_mod.generate(state["results"], state.get("feedback"), _llm(state), ctx)
    return {"scheme": out.model_dump(), "context": ctx,
            "trace": [ev("tool_scheme_gen", t0, {"feedback": state.get("feedback")}, out.model_dump(),
                         f"上下文 {ctx['total']}/{ctx['budget']} tokens，较全量节省 {ctx['saved_pct']}%")]}


def verify_node(state):
    t0 = time.perf_counter()
    c = _llm(state)
    v = verify_mod.check(state["scheme"], state["results"], c.fact_check if c else None)
    return {"verify": v, "trace": [ev("verify_node", t0, None, v, f"置信度 {v['confidence']}",
                                      "ok" if v["passed"] else "error")]}


def retry_node(state):
    t0 = time.perf_counter()
    errs = state["verify"]["errors"]
    return {"retries": state.get("retries", 0) + 1, "feedback": errs,
            "trace": [ev("retry_node", t0, None, errs, f"第 {state.get('retries', 0) + 1} 次自我修正")]}


def human_review(state):
    t0 = time.perf_counter()
    reasons = ((state.get("verify") or {}).get("errors")
               or [e.get("message", "") for e in state.get("error_feedback") or []] or ["无法自动处理"])
    final = {"outcome": "HUMAN", "text": "这个问题我暂时无法给出可靠答复，已为您转接人工客服。",
             "reasons": reasons, "options": [], "draft": state.get("scheme")}
    return {"outcome": "HUMAN", "final": final,
            "trace": [ev("human_review", t0, None, reasons, "转人工", "human")]}


def finish(state):
    t0 = time.perf_counter()
    outcome = state.get("outcome") or "ANSWER"
    p = ParsedQuery.model_validate(state["parsed"]) if state.get("parsed") else None
    if outcome == "ANSWER":
        sc = state["scheme"]
        final = {"outcome": "ANSWER", "text": sc["text"], "options": sc["options"], "verify": state["verify"]}
    else:
        final = state["final"]
    mem = state["memory"]
    if p and p.order_id:
        memory.push_order(mem, p.order_id)
    c = _llm(state)
    memory.add_turn(mem, "user", state["question"], c)
    memory.add_turn(mem, "assistant", final["text"], c)
    if p and p.sub_queries:
        mem["last_intent"] = p.sub_queries[-1].intent
    memory.save_session(state["session_id"], state["user_id"], mem)
    if outcome == "ANSWER" and p and p.order_id:
        order = data.get_order(p.order_id)
        for i, r in enumerate(state["results"]):
            if r["intent"] in ("CHANGE", "REFUND"):
                memory.update_profile(state["user_id"], order, r["intent"],
                                      key=f"{state['session_id']}:{len(mem['turns'])}:{len(mem.get('archive', []))}:{i}")
    return {"outcome": outcome, "final": final, "memory": mem,
            "trace": [ev("finish", t0, None, {"outcome": outcome}, "写入记忆，本轮结束")]}


def _actionable(final):
    return [i for i, o in enumerate(final.get("options", []))
            if o.get("allowed") and o.get("type") in ("CHANGE", "REFUND")]


def await_confirm(state):
    t0 = time.perf_counter()
    final = state["final"]
    idx = _actionable(final)
    answer = interrupt({"kind": "confirm", "order_id": state["parsed"]["order_id"],
                        "options": [{"index": i, **final["options"][i]} for i in idx],
                        "message": "确认后将立即办理，操作不可撤销。"})
    if not (isinstance(answer, dict) and answer.get("confirm")):
        return {"outcome": "CANCELLED", "final": {**final, "outcome": "CANCELLED"},
                "trace": [ev("await_confirm", t0, None, answer, "乘客取消办理")]}
    i = int(answer.get("index", idx[0]))
    if i not in idx:
        return {"outcome": "CANCELLED", "final": {**final, "outcome": "CANCELLED"},
                "trace": [ev("await_confirm", t0, None, answer, "选择的方案不可办理", "error")]}
    confirm = {"index": i, "confirm_id": answer.get("confirm_id") or f"c-{uuid.uuid4().hex[:12]}"}
    return {"confirm": confirm, "trace": [ev("await_confirm", t0, None, confirm, "乘客已确认")]}


def tool_issue_ticket(state):
    t0 = time.perf_counter()
    cf = state["confirm"]
    r = state["results"][cf["index"]]
    args = {"order_id": state["parsed"]["order_id"], "action": r["intent"], "confirm_id": cf["confirm_id"]}
    if r["intent"] == "CHANGE":
        args.update(target_date=r["target_date"], target_flight=r["fee"]["target"]["flight_no"])
    res = tools.issue_ticket(args)
    if res["status"] == "CHANGED":
        text = f"已为您改签至 {res['target']['date']} {res['target']['flight_no']}，新票号 {res['ticket_no']}，共支付{res['amount']}元。"
    elif res["status"] == "REFUNDED":
        text = f"退票已受理，预计退款{res['amount']}元，将原路退回。"
    else:
        text = f"办理失败：{res['reason']}"
    outcome = "TICKETED" if res["status"] in ("CHANGED", "REFUNDED") else "HUMAN"
    return {"outcome": outcome, "final": {**state["final"], "outcome": outcome, "text": text, "ticket": res},
            "trace": [ev("tool_issue_ticket", t0, args, res, text, "ok" if outcome == "TICKETED" else "error")]}


# ---------------- 路由 ----------------
def after_understand(state):
    return "clarify" if state["parsed"].get("clarify") else "agent_node"


def after_clarify(state):
    return "human_review" if state.get("outcome") == "HUMAN" else "query_understand"


def after_agent(state):
    a = state["next_action"]["action"]
    if a == "finish":
        return "sub_done"
    if a in ("clarify", "human"):
        return {"clarify": "clarify", "human": "human_review"}[a]
    return "schema_validate"           # 未知工具名也交给校验节点报 UNKNOWN_TOOL


def after_validate(state):
    if state.get("error_feedback"):
        return "human_review" if state["tool_retries"] > S.get()["planner"]["max_tool_retries"] else "agent_node"
    return "checkpoint_gate"


def after_gate(state):
    if state.get("gate_reason"):
        return "human_review" if state.get("gate_blocks", 0) > 2 else "agent_node"
    return "tool_exec"


def after_tool(state):
    last = state["calls"][-1]
    if not last["ok"]:
        return "sub_done"
    p = state["parsed"]
    if (last["tool"] == "tool_rule_retrieve" and not state.get("stepped_back") and (p.get("fare_product") or p.get("airline"))
            and len(last["output"]["rules"]) < S.get()["query"]["step_back_min_hits"]):
        return "step_back"
    return "agent_node"


def after_sub(state):
    return "agent_node" if state["sub_idx"] < len(state["parsed"]["sub_queries"]) else "tool_scheme_gen"


def after_verify(state):
    if state["verify"]["passed"]:
        return "finish"
    return "retry_node" if state.get("retries", 0) < S.get()["verify"]["max_retries"] else "human_review"


def after_finish(state):
    return "await_confirm" if state["outcome"] == "ANSWER" and _actionable(state["final"]) else END


def after_confirm(state):
    return "tool_issue_ticket" if state.get("confirm") and state.get("outcome") != "CANCELLED" else END


def build_graph(checkpointer):
    b = StateGraph(AgentState)
    for fn in (user_input, memory_node, order_align, query_understand, clarify, agent_node, schema_validate,
               checkpoint_gate, tool_exec, step_back, sub_done, tool_scheme_gen, verify_node, retry_node,
               human_review, finish, await_confirm, tool_issue_ticket):
        b.add_node(fn.__name__, fn)
    b.add_edge(START, "user_input")
    b.add_edge("user_input", "memory_node")
    b.add_edge("memory_node", "order_align")
    b.add_edge("order_align", "query_understand")
    b.add_conditional_edges("query_understand", after_understand, ["clarify", "agent_node"])
    b.add_conditional_edges("clarify", after_clarify, ["human_review", "query_understand"])
    b.add_conditional_edges("agent_node", after_agent, ["sub_done", "clarify", "human_review", "schema_validate"])
    b.add_conditional_edges("schema_validate", after_validate, ["human_review", "agent_node", "checkpoint_gate"])
    b.add_conditional_edges("checkpoint_gate", after_gate, ["human_review", "agent_node", "tool_exec"])
    b.add_conditional_edges("tool_exec", after_tool, ["sub_done", "step_back", "agent_node"])
    b.add_edge("step_back", "agent_node")
    b.add_conditional_edges("sub_done", after_sub, ["agent_node", "tool_scheme_gen"])
    b.add_edge("tool_scheme_gen", "verify_node")
    b.add_conditional_edges("verify_node", after_verify, ["finish", "retry_node", "human_review"])
    b.add_edge("retry_node", "tool_scheme_gen")
    b.add_edge("human_review", "finish")
    b.add_conditional_edges("finish", after_finish, ["await_confirm", END])
    b.add_conditional_edges("await_confirm", after_confirm, ["tool_issue_ticket", END])
    b.add_edge("tool_issue_ticket", END)
    return b.compile(checkpointer=checkpointer)


# ---------------- 驱动 ----------------
def get_graph():
    global _graph
    if _graph is None:
        S.VAR_DIR.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(S.VAR_DIR / "checkpoints.db", check_same_thread=False)
        _graph = build_graph(SqliteSaver(conn))
    return _graph


def reset_graph():
    global _graph
    _graph = None
    _clients.clear()


def _cfg(session_id):
    return {"configurable": {"thread_id": session_id}, "recursion_limit": 80}


def pending_interrupt(session_id):
    snap = get_graph().get_state(_cfg(session_id))
    ints = list(getattr(snap, "interrupts", ()) or [i for t in snap.tasks for i in t.interrupts])
    return ints[0].value if ints else None


def _drive(inp, session_id, on_event=None):
    g = get_graph()
    events, pending = [], None
    for chunk in g.stream(inp, _cfg(session_id), stream_mode="updates"):
        for node, upd in chunk.items():
            if node == "__interrupt__":
                pending = upd[0].value
                if on_event:
                    on_event({"type": "interrupt", "payload": pending})
                continue
            for e in (upd or {}).get("trace", []):
                events.append(e)
                if on_event:
                    on_event({"type": "trace", "event": e})
    st = g.get_state(_cfg(session_id)).values
    mem = st.get("memory") or {}
    return {"session_id": session_id, "outcome": st.get("outcome"), "final": st.get("final"),
            "interrupt": pending, "trace": events, "parsed": st.get("parsed"), "context": st.get("context"),
            "results": st.get("results"), "verify": st.get("verify"),
            "memory": {k: mem.get(k) for k in ("order_stack", "summary", "turns", "last_intent")}}


def run_turn(session_id, user_id, message, order_id=None, mode=None, client=None, on_event=None):
    mode = mode or S.get()["mode"]
    _clients[session_id] = client if client is not None else llm_mod.get_client(mode)
    inp = {"session_id": session_id, "user_id": user_id, "question": message, "order_id": order_id, "mode": mode}
    return _drive(inp, session_id, on_event)


def resume(session_id, payload, client=None, on_event=None):
    if client is not None:
        _clients[session_id] = client
    elif session_id not in _clients:     # 服务重启后：按检查点里的模式重建客户端
        mode = get_graph().get_state(_cfg(session_id)).values.get("mode", "offline")
        _clients[session_id] = llm_mod.get_client(mode)
    return _drive(Command(resume=payload), session_id, on_event)
```

- [ ] **Step 6：运行测试，确认通过**

Run: `uv run pytest tests/test_graph.py tests/test_graph_llm.py -v`
Expected: 15 passed

排查提示：
- 如果 `test_resume_after_restart` 失败，先确认 `get_state(...).interrupts` 在当前 LangGraph 版本里的位置，再改 `pending_interrupt()`，不要绕过检查点。
- 如果 `test_off_topic_goes_human` 没有转人工，先看 `tool_exec` 的 trace 是否命中了规则，回到 Task 5 调 `dense_min`。

- [ ] **Step 7：跑全量测试并提交**

Run: `uv run pytest -v`
Expected: 全部 passed（slow 测试默认跳过）

```bash
git add air_agent/state.py air_agent/graph.py tests/test_graph.py tests/test_graph_llm.py
git commit -m "feat: wire LangGraph state machine with interrupts and checkpoints"
```

### Task 12：金标准评测（构建、校验、运行、指标）

**Files:**
- Create: `air_agent/evaluation.py`、`eval/gold_cases.json`（由脚本生成后提交）
- Test: `tests/test_evaluation.py`

**评测隔离**：评测固定在 `2026-09-26T10:00` 运行，使用独立的沙箱库 `var/eval_sandbox/`，每条用例开始前重置数据（`data.init(force=True)`），防止画像累积影响检索。阶段二的 API 通过子进程调用 CLI，所以不会影响正在演示的对话。

**订单引用**：用例里写 `"#1"`～`"#12"`，运行时换算成 `ORD20260926001` 这样的订单号。

**期望金额**：由构建脚本调用运价引擎生成，校验时用引擎重算一次。评测要检验的是：Agent 是否拿到了正确的引擎结果，并原样写进方案。README 里要写明这一点。

- [ ] **Step 1：写失败测试 `tests/test_evaluation.py`**

```python
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
```

- [ ] **Step 2：运行测试，确认失败**

Run: `uv run pytest tests/test_evaluation.py -v`
Expected: FAIL，`ImportError: cannot import name 'evaluation'`

- [ ] **Step 3：实现 `air_agent/evaluation.py`**

```python
"""金标准评测：构建用例、校验用例、批量运行、计算指标。

CLI（在独立沙箱里运行，不影响演示数据）：
  uv run python -m air_agent.evaluation build              # 生成 eval/gold_cases.json
  uv run python -m air_agent.evaluation validate           # 校验用例，输出 JSON
  uv run python -m air_agent.evaluation run --mode offline # 运行并写入 var/eval_runs/<run_id>.json
"""
import argparse
import json
import shutil
import statistics
import sys
import time
import uuid
from collections import Counter, defaultdict
from datetime import date, datetime, timedelta
from typing import Literal
from pydantic import BaseModel, Field, ValidationError
from . import data, engine, graph, tools
from . import settings as S

EVAL_NOW = "2026-09-26T10:00"
CASES_PATH = S.ROOT / "eval" / "gold_cases.json"
RUNS_DIR = S.ROOT / "var" / "eval_runs"
REACT = ["tool_flight_status", "tool_rule_retrieve", "tool_recalc_fee"]
FORBID = ["quote_mismatch", "unsupported_claim", "skip_rule"]


class Expect(BaseModel):
    intents: list[Literal["CHANGE", "REFUND", "POLICY", "STATUS"]]
    tools: list[str]
    fees: dict[str, int | None] = Field(default_factory=dict)
    rule_ids: list[str] = Field(default_factory=list)
    outcome: Literal["ANSWER", "HUMAN", "CLARIFY"]
    schema_errors: list[str] = Field(default_factory=list)


class Turn(BaseModel):
    message: str
    order: str | None = None


class Case(BaseModel):
    id: str = Field(pattern=r"^[\w-]{1,64}$")
    title: str
    tags: list[str] = Field(default_factory=list)
    user_id: str
    airline: str | None = None
    fare_product: str | None = None
    order: str | None = None
    question: str = Field(min_length=1, max_length=1000)
    history: list[Turn] = Field(default_factory=list)
    clarify_answers: list[str] = Field(default_factory=list)
    expect: Expect
    forbid: list[str] = Field(default_factory=lambda: list(FORBID))


# ---------------- 订单引用 ----------------
def order_id(ref: str | None) -> str | None:
    if not ref:
        return None
    if ref.startswith("#"):
        return f"ORD{datetime.fromisoformat(EVAL_NOW):%Y%m%d}{int(ref[1:]):03d}"
    return ref


def _md(d: str) -> str:
    x = date.fromisoformat(d)
    return f"{x.month}月{x.day}号"


def _fee_expect(oid, intent, target=None):
    r = engine.calc(oid, intent, target_date=target)
    if not r["allowed"]:
        return None, r["rule_id"]
    return (r["total"] if intent == "CHANGE" else r["refund_amount"]), r["rule_id"]


# ---------------- 构建 ----------------
def build_cases() -> list[dict]:
    """按 12 张演示订单 × 4 类问法 + 政策 / 追问 / 指代 / 转人工 生成初始用例。"""
    data.init(force=True)
    cases = []
    full = ["tool_flight_status", "tool_rule_retrieve", "tool_recalc_fee"]
    for n in range(1, 13):
        ref = f"#{n}"
        o = data.get_order(order_id(ref))
        f = data.get_flight(o["flight_no"], o["date"])
        future = engine.departure(f) > S.now()
        target = (max(date.fromisoformat(o["date"]), S.now().date()) + timedelta(days=1)).isoformat()
        base_tags = []
        if engine.involuntary_reason(f):
            base_tags.append("非自愿")
        if not future:
            base_tags.append("误机")
        chg, chg_rule = _fee_expect(o["order_id"], "CHANGE", target)
        rfd, rfd_rule = _fee_expect(o["order_id"], "REFUND")
        common = {"user_id": o["user_id"], "airline": o["airline"], "fare_product": o["fare_product"], "order": ref}
        variants = [
            ("chg", f"帮我把这张票改到{_md(target)}", ["CHANGE"], full, {"CHANGE": chg}, [chg_rule], ["单诉求"]),
            ("rfd", "这张票退掉能退多少钱", ["REFUND"], full, {"REFUND": rfd}, [rfd_rule], ["单诉求"]),
            ("both", f"改到{_md(target)}要多少钱，顺便看看退票多少", ["CHANGE", "REFUND"], full + full,
             {"CHANGE": chg, "REFUND": rfd}, sorted({chg_rule, rfd_rule}), ["多诉求"]),
            ("status", "这个航班现在什么情况", ["STATUS"], ["tool_flight_status"], {}, [], ["航班状态"]),
        ]
        for key, q, intents, tl, fees, rules, tags in variants:
            extra = ["不可办理"] if any(v is None for v in fees.values()) else []
            cases.append({"id": f"o{n:02d}-{key}", "title": f"订单#{n} {o['airline']} {o['fare_product']} {q}",
                          "tags": tags + base_tags + extra, **common, "question": q, "history": [],
                          "clarify_answers": [],
                          "expect": {"intents": intents, "tools": tl, "fees": fees, "rule_ids": rules,
                                     "outcome": "ANSWER", "schema_errors": []}, "forbid": list(FORBID)})
    policy = [("MU", "SPECIAL", "东方云航特价票能退吗", "MU-SPECIAL-RFD"),
              ("CA", "F", "国际星航头等舱改签收费吗", "CA-F-CHG"),
              ("CZ", "Y_DISCOUNT", "南方海航折扣票退票手续费多少", "CZ-Y_DISCOUNT-RFD"),
              ("HU", "GROUP", "海岛航空团队票能改签吗", "HU-GROUP-CHG"),
              ("MU", "C", "东方云航公务舱退票怎么收费", "MU-C-RFD"),
              ("CA", "Y", "国际星航经济舱全价改期费多少", "CA-Y-CHG"),
              ("CZ", "F", "南方海航头等舱退票", "CZ-F-RFD"),
              ("HU", "Y_DISCOUNT", "海岛航空打折票改签", "HU-Y_DISCOUNT-CHG")]
    for i, (al, prod, q, rid) in enumerate(policy, 1):
        cases.append({"id": f"policy-{i:02d}", "title": q, "tags": ["政策咨询"], "user_id": "u_wang",
                      "airline": al, "fare_product": prod, "order": None, "question": q, "history": [],
                      "clarify_answers": [],
                      "expect": {"intents": ["POLICY"], "tools": ["tool_rule_retrieve"], "fees": {}, "rule_ids": [rid],
                                 "outcome": "ANSWER", "schema_errors": []}, "forbid": list(FORBID)})
    o1, o3 = data.get_order(order_id("#1")), data.get_order(order_id("#3"))
    r3, r3_rule = _fee_expect(o3["order_id"], "REFUND")
    c1, c1_rule = _fee_expect(o1["order_id"], "CHANGE", "2026-09-29")
    c1b, _ = _fee_expect(o1["order_id"], "CHANGE", "2026-10-02")
    r1, r1_rule = _fee_expect(o1["order_id"], "REFUND")
    o2 = data.get_order(order_id("#2"))
    r2, r2_rule = _fee_expect(o2["order_id"], "REFUND")
    c3, c3_rule = _fee_expect(o3["order_id"], "CHANGE", "2026-10-08")
    mk = lambda **kw: {"tags": [], "user_id": "u_zhang", "airline": None, "fare_product": None, "order": None,
                       "history": [], "clarify_answers": [], "forbid": list(FORBID), **kw}
    cases += [
        mk(id="clarify-order", title="没指定订单，追问后选择订单#3", tags=["追问"], airline="CA", fare_product="C",
           question="我要退票", clarify_answers=["#3"],
           expect={"intents": ["REFUND"], "tools": full, "fees": {"REFUND": r3}, "rule_ids": [r3_rule],
                   "outcome": "ANSWER", "schema_errors": []}),
        mk(id="clarify-date", title="改签没说日期，追问后选 9月29日", tags=["追问"], airline="MU",
           fare_product="Y_DISCOUNT", order="#1", question="我想改签", clarify_answers=["2026-09-29"],
           expect={"intents": ["CHANGE"], "tools": full, "fees": {"CHANGE": c1}, "rule_ids": [c1_rule],
                   "outcome": "ANSWER", "schema_errors": []}),
        mk(id="clarify-vague-date", title="模糊日期触发 Schema 校验，再追问", tags=["追问", "Schema"], airline="MU",
           fare_product="Y_DISCOUNT", order="#1", question="改到下个月初", clarify_answers=["2026-10-02"],
           expect={"intents": ["CHANGE"], "tools": full, "fees": {"CHANGE": c1b}, "rule_ids": [c1_rule],
                   "outcome": "ANSWER", "schema_errors": ["TYPE"]}),
        mk(id="coref-return", title="上一轮提到去程，这一轮问回程", tags=["指代"], airline="MU", fare_product="Y_DISCOUNT",
           history=[{"message": "帮我把杭州飞北京那张改到9月29号", "order": None}], question="回程那张能退吗",
           expect={"intents": ["REFUND"], "tools": full, "fees": {"REFUND": r2}, "rule_ids": [r2_rule],
                   "outcome": "ANSWER", "schema_errors": []}),
        mk(id="coref-ellipsis", title="省略追问：那退票呢", tags=["指代"], airline="MU", fare_product="Y_DISCOUNT",
           history=[{"message": "改到9月29号", "order": "#1"}], question="那退票呢",
           expect={"intents": ["REFUND"], "tools": full, "fees": {"REFUND": r1}, "rule_ids": [r1_rule],
                   "outcome": "ANSWER", "schema_errors": []}),
        mk(id="coref-stack", title="订单栈：上一轮问#3，这一轮只说日期", tags=["指代"], airline="CA", fare_product="C",
           history=[{"message": "退票多少钱", "order": "#3"}], question="那改到10月8号呢",
           expect={"intents": ["CHANGE"], "tools": full, "fees": {"CHANGE": c3}, "rule_ids": [c3_rule],
                   "outcome": "ANSWER", "schema_errors": []}),
    ]
    for i, q in enumerate(["宠物能带上飞机吗", "会员积分怎么兑换里程", "机上餐食可以选素食吗"], 1):
        cases.append(mk(id=f"human-{i:02d}", title=q, tags=["转人工"], user_id="u_wang", question=q,
                        expect={"intents": ["POLICY"], "tools": ["tool_rule_retrieve"], "fees": {}, "rule_ids": [],
                                "outcome": "HUMAN", "schema_errors": []}))
    return cases


# ---------------- 校验 ----------------
def validate_cases(cases: list[dict]) -> dict[str, list[dict]]:
    """返回 {case_id: [{field, message}]}；没有错误的用例不出现。"""
    data.init(force=True)
    out: dict[str, list[dict]] = defaultdict(list)
    seen = Counter(c.get("id") for c in cases)
    rule_ids = {r["id"] for r in data.list_rules()}
    users = {p["user_id"] for p in data.list_profiles()}
    for raw in cases:
        cid = raw.get("id") or "?"
        try:
            c = Case.model_validate(raw)
        except ValidationError as e:
            for err in e.errors():
                out[cid].append({"field": ".".join(str(x) for x in err["loc"]), "message": err["msg"]})
            continue
        if seen[c.id] > 1:
            out[cid].append({"field": "id", "message": "用例 ID 重复"})
        if c.user_id not in users:
            out[cid].append({"field": "user_id", "message": f"用户不存在：{c.user_id}"})
        oid = order_id(c.order)
        if c.order and not data.get_order(oid):
            out[cid].append({"field": "order", "message": f"订单不存在：{c.order}"})
        for i, h in enumerate(c.history):
            if h.order and not data.get_order(order_id(h.order)):
                out[cid].append({"field": f"history.{i}.order", "message": f"订单不存在：{h.order}"})
        bad_tools = [t for t in c.expect.tools if t not in REACT]
        if bad_tools:
            out[cid].append({"field": "expect.tools", "message": f"非法工具：{bad_tools}"})
        bad_rules = [r for r in c.expect.rule_ids if r not in rule_ids]
        if bad_rules:
            out[cid].append({"field": "expect.rule_ids", "message": f"规则不存在：{bad_rules}"})
        fee_order = oid or order_id(next((a for a in c.clarify_answers if a.startswith("#")), None)) \
            or _last_history_order(c)
        for intent, val in c.expect.fees.items():
            if intent not in ("CHANGE", "REFUND"):
                out[cid].append({"field": f"expect.fees.{intent}", "message": "只能是 CHANGE 或 REFUND"})
                continue
            if not fee_order or not data.get_order(fee_order):
                out[cid].append({"field": f"expect.fees.{intent}", "message": "无法确定订单，不能校验金额"})
                continue
            target = _case_target(c) if intent == "CHANGE" else None
            try:
                got, _ = _fee_expect(fee_order, intent, target)
            except engine.EngineError as e:
                out[cid].append({"field": f"expect.fees.{intent}", "message": f"引擎报错：{e}"})
                continue
            if got != val:
                out[cid].append({"field": f"expect.fees.{intent}", "message": f"期望金额与引擎不符，引擎值为 {got}"})
    return dict(out)


def _last_history_order(c: Case):
    """指代类用例：订单来自历史或代词，取能推出的订单（与 build_cases 的写法一致）。"""
    from . import query
    ctx = {"user_id": c.user_id, "selected_order_id": None, "order_stack": [], "last_intent": None}
    for h in c.history:
        ctx["selected_order_id"] = order_id(h.order)
        oid = query.resolve_order(h.message, ctx)
        if oid:
            ctx["order_stack"] = [oid] + ctx["order_stack"]
        ctx["selected_order_id"] = None
    return query.resolve_order(c.question, ctx)


def _case_target(c: Case):
    from . import query
    for a in c.clarify_answers:
        if not a.startswith("#"):
            return a
    t = query.find_date(c.question)
    return t if t and t[:2] == "20" else None


# ---------------- 运行 ----------------
def run_case(case: dict, mode: str, run_id: str, client=None) -> dict:
    c = Case.model_validate(case)
    data.init(force=True)
    sid = f"eval-{run_id}-{c.id}"
    for h in c.history:
        graph.run_turn(sid, c.user_id, h.message, order_id=order_id(h.order), mode=mode, client=client)
    t0 = time.perf_counter()
    res = graph.run_turn(sid, c.user_id, c.question, order_id=order_id(c.order), mode=mode, client=client)
    answers = list(c.clarify_answers)
    trace = list(res["trace"])
    while res["interrupt"] and res["interrupt"]["kind"] == "clarify" and answers:
        a = answers.pop(0)
        res = graph.resume(sid, {"value": order_id(a) if a.startswith("#") else a}, client=client)
        trace += res["trace"]
    ms = round((time.perf_counter() - t0) * 1000, 1)
    outcome = "CLARIFY" if res["interrupt"] and res["interrupt"]["kind"] == "clarify" else res["outcome"]
    results = res["results"] or []
    final = res["final"] or {}
    tools_seq = [x["tool"] for r in results for x in r["calls"] if x["ok"] and x["tool"] in REACT]
    schema_errs = [e["output"][0]["kind"] for e in trace if e["node"] == "schema_validate" and e["status"] == "error"]
    blocked = any(e["node"] == "checkpoint_gate" and e["status"] == "blocked" for e in trace)
    retrieved = {x["id"] for r in results for x in r["rules"]}
    fees = {}
    for opt, r in zip(final.get("options", []), results):
        if r["intent"] in ("CHANGE", "REFUND"):
            fees[r["intent"]] = opt["total"] if opt.get("allowed") else None
    v = res["verify"] or {}
    reasons = []
    tool_ok = tools_seq == c.expect.tools and all(k in c.expect.schema_errors for k in schema_errs)
    if tools_seq != c.expect.tools:
        reasons.append("wrong_tools")
    reasons += [k for k in schema_errs if k not in c.expect.schema_errors]
    hit = set(c.expect.rule_ids) <= retrieved if c.expect.rule_ids else None
    if hit is False:
        reasons.append("rule_miss")
    if outcome != c.expect.outcome:
        reasons.append("wrong_outcome")
    if outcome == "ANSWER" and [r["intent"] for r in results] != c.expect.intents:
        reasons.append("wrong_intent")
    if outcome == "ANSWER" and any(fees.get(k) != val for k, val in c.expect.fees.items()):
        reasons.append("fee_mismatch")
    if "skip_rule" in c.forbid and blocked:
        reasons.append("skip_rule")
    if outcome == "ANSWER" and "quote_mismatch" in c.forbid and v.get("quote_mismatch"):
        reasons.append("quote_mismatch")
    if outcome == "ANSWER" and "unsupported_claim" in c.forbid and v.get("unsupported_claims"):
        reasons.append("unsupported_claim")
    passed = not [r for r in reasons if r not in ("wrong_tools", "rule_miss", *c.expect.schema_errors)] \
        and outcome == c.expect.outcome
    tokens = (res.get("context") or {}).get("total", 0)
    return {"case_id": c.id, "title": c.title, "tags": c.tags,
            "group": f"{c.airline or '通用'}·{c.fare_product or '通用'}", "passed": passed, "tool_ok": tool_ok,
            "policy_hit": hit, "reasons": reasons, "ms": ms, "tokens": tokens, "expected": c.expect.model_dump(),
            "actual": {"tools": tools_seq, "intents": [r["intent"] for r in results], "fees": fees,
                       "outcome": outcome, "rule_ids": sorted(retrieved), "text": final.get("text", ""),
                       "schema_errors": schema_errs},
            "trace": trace}


def summarize(results: list[dict]) -> dict:
    n = len(results) or 1
    lat = sorted(r["ms"] for r in results) or [0]
    hits = [r["policy_hit"] for r in results if r["policy_hit"] is not None]
    groups = defaultdict(lambda: {"n": 0, "passed": 0})
    for r in results:
        groups[r["group"]]["n"] += 1
        groups[r["group"]]["passed"] += r["passed"]
    return {"n": len(results),
            "tool_acc": round(sum(r["tool_ok"] for r in results) / n, 4),
            "policy_hit": round(sum(hits) / len(hits), 4) if hits else None,
            "correctness": round(sum(r["passed"] for r in results) / n, 4),
            "human_rate": round(sum(r["actual"]["outcome"] == "HUMAN" for r in results) / n, 4),
            "latency_avg_ms": round(statistics.mean(lat), 1),
            "latency_p95_ms": lat[min(len(lat) - 1, int(len(lat) * 0.95))],
            "tokens_avg": round(statistics.mean([r["tokens"] for r in results]), 1) if results else 0,
            "failure_reasons": dict(Counter(x for r in results for x in r["reasons"])),
            "by_group": {k: {**v, "rate": round(v["passed"] / v["n"], 4)} for k, v in sorted(groups.items())}}


def run_cases(cases: list[dict], mode: str = "offline", run_id: str | None = None, on_progress=None, client=None):
    run_id = run_id or f"{S.now():%Y%m%d%H%M}-{uuid.uuid4().hex[:6]}"
    results = []
    for i, c in enumerate(cases, 1):
        try:
            r = run_case(c, mode, run_id, client)
        except Exception as e:      # 单条用例出错不影响整批，记为失败
            r = {"case_id": c.get("id"), "title": c.get("title", ""), "tags": c.get("tags", []), "group": "错误",
                 "passed": False, "tool_ok": False, "policy_hit": None, "reasons": [f"error:{type(e).__name__}"],
                 "ms": 0, "tokens": 0, "expected": c.get("expect"), "actual": {"outcome": "ERROR", "text": str(e)},
                 "trace": []}
        results.append(r)
        if on_progress:
            on_progress({"type": "progress", "done": i, "total": len(cases), "case_id": r["case_id"],
                         "passed": r["passed"]})
    return {"id": run_id, "created_at": datetime.now().isoformat(timespec="seconds"), "mode": mode,
            "config": S.get(), "summary": summarize(results), "results": results}


# ---------------- CLI ----------------
def _sandbox():
    sandbox = S.ROOT / "var" / "eval_sandbox"
    shutil.rmtree(sandbox, ignore_errors=True)
    S.VAR_DIR = sandbox
    S.set_now(datetime.fromisoformat(EVAL_NOW))
    graph.reset_graph()


def _emit(obj):
    sys.stdout.write(json.dumps(obj, ensure_ascii=False) + "\n")
    sys.stdout.flush()


def main(argv=None):
    ap = argparse.ArgumentParser(prog="air_agent.evaluation")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("build")
    v = sub.add_parser("validate")
    v.add_argument("--cases", default=str(CASES_PATH))
    r = sub.add_parser("run")
    r.add_argument("--cases", default=str(CASES_PATH))
    r.add_argument("--mode", default="offline", choices=["offline", "deepseek"])
    r.add_argument("--ids", default="")
    r.add_argument("--run-id", default=None)
    args = ap.parse_args(argv)
    runs_dir = RUNS_DIR
    _sandbox()
    if args.cmd == "build":
        CASES_PATH.parent.mkdir(parents=True, exist_ok=True)
        cases = build_cases()
        CASES_PATH.write_text(json.dumps(cases, ensure_ascii=False, indent=2), "utf-8")
        _emit({"type": "built", "count": len(cases), "path": str(CASES_PATH)})
        return
    cases = json.loads(open(args.cases, encoding="utf-8").read())
    if args.cmd == "validate":
        _emit({"type": "validated", "errors": validate_cases(cases)})
        return
    if args.ids:
        keep = set(args.ids.split(","))
        cases = [c for c in cases if c["id"] in keep]
    run = run_cases(cases, args.mode, args.run_id, on_progress=_emit)
    runs_dir.mkdir(parents=True, exist_ok=True)
    (runs_dir / f"{run['id']}.json").write_text(json.dumps(run, ensure_ascii=False), "utf-8")
    _emit({"type": "done", "run_id": run["id"], "summary": run["summary"]})


if __name__ == "__main__":
    main()
```

- [ ] **Step 4：运行测试，确认通过**

Run: `uv run pytest tests/test_evaluation.py -v`
Expected: 7 passed

如果 `test_full_gold_offline_baseline` 低于 0.9：按断言里输出的失败用例逐条看 `reasons`，属于实现缺陷的（例如意图识别漏词、指代没解析出来）回到对应模块修代码并补单元测试；不要通过改用例期望来凑分。`rule_miss` 和 `wrong_tools` 不计入正确率，只计入政策命中率和工具调用准确率。

- [ ] **Step 5：生成并校验金标准文件**

Run:
```bash
uv run python -m air_agent.evaluation build
uv run python -m air_agent.evaluation validate
uv run python -m air_agent.evaluation run --mode offline | tail -1
```
Expected：
- build 输出 `{"type": "built", "count": 65, ...}`；
- validate 输出 `{"type": "validated", "errors": {}}`；
- run 最后一行是 `done`，其中 `correctness` ≥ 0.9。

把 run 输出的 summary 记下来，写进 Task 13 的 README。

- [ ] **Step 6：提交**

```bash
git add air_agent/evaluation.py eval/gold_cases.json tests/test_evaluation.py
git commit -m "feat: add gold-standard evaluation (build/validate/run/metrics)"
```

### Task 13：启动脚本、README 与阶段验收

**Files:**
- Create: `start.sh`、`config.example.json`、`README.md`、`air_agent/__main__.py`

阶段一还没有 Web 服务，`start.sh` 先提供命令行对话，方便验收；阶段二加入 FastAPI 后改成启动服务。

- [ ] **Step 1：写 `air_agent/__main__.py`（命令行对话）**

```python
"""命令行体验：uv run python -m air_agent --user u_zhang --order ORD...001"""
import argparse
import json
import uuid
from . import data, graph
from . import settings as S


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--user", default="u_zhang")
    ap.add_argument("--order", default=None)
    ap.add_argument("--mode", default=None, choices=[None, "offline", "deepseek"])
    args = ap.parse_args()
    data.init()
    sid = f"cli-{uuid.uuid4().hex[:8]}"
    print("订单：")
    for o in data.list_orders(args.user):
        print(" ", o["order_id"], o["flight_no"], o["date"], o["fare_product"], o["status"])
    print(f"模式：{args.mode or S.get()['mode']}。输入问题，空行退出。")
    pending = None
    while True:
        q = input("\n你：").strip()
        if not q:
            break
        if pending and pending["kind"] == "clarify":
            res = graph.resume(sid, {"value": q})
        elif pending and pending["kind"] == "confirm" and q in ("确认", "y"):
            res = graph.resume(sid, {"confirm": True, "index": pending["options"][0]["index"]})
        else:
            res = graph.run_turn(sid, args.user, q, order_id=args.order, mode=args.mode)
        for e in res["trace"]:
            print(f"  [{e['status']:>7}] {e['node']:<16} {e['note']}")
        pending = res["interrupt"]
        if res.get("final", {}).get("text"):
            print("助手：", res["final"]["text"])
        if pending:
            print("等待：", json.dumps({k: pending[k] for k in pending if k != "options"}, ensure_ascii=False))
            for opt in pending.get("options", []):
                print("   -", opt.get("label") or opt.get("title"), opt.get("value") or opt.get("total") or "")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2：写 `start.sh` 与 `config.example.json`**

```bash
#!/usr/bin/env bash
# 机票退改签 Agent：安装依赖并启动（阶段一为命令行对话）
set -euo pipefail
cd "$(dirname "$0")"
command -v uv >/dev/null || { echo "请先安装 uv：https://docs.astral.sh/uv/"; exit 1; }
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
uv sync
uv run python -m air_agent "$@"
```

```json
{
  "mode": "offline",
  "llm": {"base_url": "https://api.deepseek.com", "api_key": "", "model": "deepseek-chat", "timeout": 30}
}
```

Run: `chmod +x start.sh`

- [ ] **Step 3：写 `README.md`**

内容包括：
1. 一句话介绍与「数据全部虚构」声明；
2. 快速启动：`./start.sh`，以及 DeepSeek 配置方法（复制 `config.example.json` 为 `config.json`，填写 `api_key`，`mode` 改为 `deepseek`）；
3. 五项工作与代码位置对照表；
4. 6 条演示脚本：改签差价、改签加退票、航班取消、指代追问、模糊日期触发 Schema 校验、无关问题转人工；
5. 评测命令和 Task 12 Step 5 的实测指标（写实测数字，并说明期望金额来自运价引擎）。

对照表：

| 工作项 | 节点 | 代码位置 |
|---|---|---|
| 一 ReAct 规划编排 | agent_node / schema_validate / checkpoint_gate / tool_exec / retry_node | `air_agent/graph.py`、`planner.py`、`tools.py` |
| 二 Query 理解 | order_align / query_understand / clarify / step_back | `air_agent/query.py` |
| 三 混合检索与精排 | tool_exec(tool_rule_retrieve) | `air_agent/retrieval.py` |
| 四 上下文与记忆 | memory_node / finish | `air_agent/memory.py` |
| 五 方案生成与质量保障 | tool_scheme_gen / verify_node / human_review / await_confirm | `air_agent/scheme.py`、`verify.py`、`evaluation.py` |

- [ ] **Step 4：阶段验收**

Run:
```bash
uv run pytest -v
uv run pytest -m slow -v
printf '帮我改到大后天，顺便看看退票多少钱\n\n' | ./start.sh --user u_zhang --order "ORD$(date +%Y%m%d)001"
```
Expected：
- 全部测试通过；
- slow 测试通过（使用真实 BGE + FlashRank）；
- 命令行输出完整的节点 trace，两个方案的金额和 `等待：{"kind": "confirm", ...}`。

注意：命令行验收使用真实的当前时间。如果当天的种子数据是在别的日期生成的，订单号前缀会不同，先运行 `uv run python -c "from air_agent import data; data.init(force=True)"` 重新生成。

- [ ] **Step 5：提交**

```bash
chmod +x start.sh
git add start.sh config.example.json README.md air_agent/__main__.py
git commit -m "feat: add CLI demo, start script and README"
```

---

## 自查记录

- **设计覆盖**：设计文档第 2～7 节都有对应任务（Task 3～12）。第 8、9 节的 API 和前端属于阶段二、三。第 10 节的错误处理：LLM 降级在 Task 9/11，引擎异常在 Task 4/11，检索降级在 Task 5，输入限制在 Task 4 的 Schema 和阶段二的 API 层。
- **与设计文档的差异**：
  - `human_review` 不再暂停等待人工，直接结束本轮并返回 outcome=HUMAN，设计文档已同步修改；
  - 评测改为在子进程沙箱中运行，避免固定时间影响演示数据；
  - 新增「绝对相关性门槛」`dense_min`，用于无关问题转人工。
- **命名一致性**：`run_turn` / `resume` / `pending_interrupt` / `reset_graph`、`tools.validate` / `run_tool` / `issue_ticket`、`query.date_options` 在各任务中的用法一致。
