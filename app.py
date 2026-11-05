"""Web 层：FastAPI 接口 + 静态前端（web/）。

启动：uv run python app.py [--port N]，只监听 127.0.0.1。
对话类接口用 SSE 推送：trace / interrupt / final / error。图在线程池里同步运行，
on_event 回调经 loop.call_soon_threadsafe 投递到 asyncio.Queue，再由响应生成器逐条发出。
"""
import argparse
import asyncio
import copy
import json
import re
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Literal

import httpx
from fastapi import FastAPI, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field, ValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException

from air_agent import data, llm, memory, retrieval, seed
from air_agent import settings as S

ROOT = Path(__file__).resolve().parent
WEB_DIR = ROOT / "web"
EVAL_CASES = ROOT / "eval" / "gold_cases.json"
ID = r"^[\w-]{1,64}$"
ID_RE = re.compile(ID)
EXECUTOR = ThreadPoolExecutor(max_workers=4, thread_name_prefix="turn")
CABIN_LABEL = {"F": "头等舱", "C": "公务舱", "Y": "经济舱", "*": "通用"}
FLIGHT_STATUS_LABEL = {"scheduled": "正常", "delayed": "延误", "cancelled": "取消"}
ORDER_STATUS_LABEL = {"ISSUED": "已出票", "CHANGED": "已改签", "REFUNDED": "已退票"}
SCENARIOS = {"VOLUNTARY_CHANGE", "VOLUNTARY_REFUND", "CANCEL", "DELAY", "NO_SHOW", "FARE_DIFF"}


# ---------------- 错误 ----------------
class ApiError(Exception):
    def __init__(self, status: int, code: str, message: str, fields: dict | None = None):
        super().__init__(message)
        self.status, self.code, self.message, self.fields = status, code, message, fields or {}


def err_body(code, message, fields=None):
    return {"error": {"code": code, "message": message, "fields": fields or {}}}


def _loc(loc) -> str:
    parts = [str(p) for p in loc if p not in ("body", "query", "path")]
    return ".".join(parts) or "body"


@asynccontextmanager
async def lifespan(_app):
    data.init()
    yield


app = FastAPI(title="退改签助手", lifespan=lifespan)


@app.exception_handler(ApiError)
async def _api_error(_req, e: ApiError):
    return JSONResponse(err_body(e.code, e.message, e.fields), status_code=e.status)


@app.exception_handler(RequestValidationError)
async def _validation_error(_req, e: RequestValidationError):
    fields = {}
    for x in e.errors():
        fields.setdefault(_loc(x["loc"]), x["msg"])
    return JSONResponse(err_body("VALIDATION", "请求参数校验失败", fields), status_code=422)


@app.exception_handler(StarletteHTTPException)
async def _http_error(_req, e: StarletteHTTPException):
    code = {404: "NOT_FOUND", 405: "METHOD_NOT_ALLOWED"}.get(e.status_code, "HTTP_ERROR")
    return JSONResponse(err_body(code, str(e.detail)), status_code=e.status_code)


@app.exception_handler(Exception)
async def _unexpected(_req, e: Exception):
    return JSONResponse(err_body("INTERNAL", f"服务内部错误：{type(e).__name__}"), status_code=500)


class Body(BaseModel):
    model_config = ConfigDict(extra="forbid")


def _check_id(value: str, field: str):
    if not ID_RE.match(value or ""):
        raise ApiError(400, "VALIDATION", f"{field} 格式不合法", {field: "只允许字母、数字、下划线和短横线，1-64 位"})


# ---------------- SSE ----------------
def sse(event: str, payload) -> str:
    return f"event: {event}\ndata: {json.dumps(payload, ensure_ascii=False, default=str)}\n\n"


SSE_HEADERS = {"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}


def _graph():
    from air_agent import graph          # 延迟导入：graph 尚未就绪时服务也能启动
    return graph


def stream_graph(call):
    """call(on_event) 在线程池里同步执行；事件经队列转成 SSE。"""
    async def gen():
        loop = asyncio.get_running_loop()
        q: asyncio.Queue = asyncio.Queue()

        def emit(evt):
            loop.call_soon_threadsafe(q.put_nowait, evt)

        def work():
            try:
                emit({"type": "final", "result": call(emit)})
            except llm.LLMNotConfigured as e:
                emit({"type": "error", "code": "LLM_NOT_CONFIGURED", "message": str(e)})
            except ImportError as e:
                emit({"type": "error", "code": "GRAPH_UNAVAILABLE", "message": f"状态图模块未就绪：{e}"})
            except Exception as e:  # 结构化错误，不把堆栈抛给页面
                emit({"type": "error", "code": "INTERNAL", "message": f"{type(e).__name__}: {e}"})
            finally:
                emit(None)

        loop.run_in_executor(EXECUTOR, work)
        while True:
            evt = await q.get()
            if evt is None:
                break
            t = evt.get("type")
            if t == "trace":
                yield sse("trace", evt.get("event"))
            elif t == "interrupt":
                yield sse("interrupt", evt.get("payload"))
            elif t == "final":
                yield sse("final", evt.get("result"))
            elif t == "error":
                yield sse("error", {"code": evt.get("code"), "message": evt.get("message")})
    return StreamingResponse(gen(), media_type="text/event-stream", headers=SSE_HEADERS)


# ---------------- 对话 ----------------
class ChatReq(Body):
    session_id: str = Field(pattern=ID)
    user_id: str = Field(pattern=ID)
    message: str = Field(min_length=1, max_length=1000)
    order_id: str | None = Field(default=None, pattern=ID)
    mode: Literal["offline", "deepseek"] | None = None


class ResumeReq(Body):
    session_id: str = Field(pattern=ID)
    payload: dict[str, Any]


@app.post("/api/chat")
def chat(req: ChatReq):
    if not data.get_profile(req.user_id):
        raise ApiError(404, "NOT_FOUND", f"用户不存在：{req.user_id}", {"user_id": "用户不存在"})
    if req.order_id:
        o = data.get_order(req.order_id)
        if not o or o["user_id"] != req.user_id:
            raise ApiError(404, "NOT_FOUND", "订单不存在或不属于该用户", {"order_id": "订单不存在或不属于该用户"})
    return stream_graph(lambda emit: _graph().run_turn(req.session_id, req.user_id, req.message.strip(),
                                                       order_id=req.order_id, mode=req.mode, on_event=emit))


@app.post("/api/resume")
def resume(req: ResumeReq):
    return stream_graph(lambda emit: _graph().resume(req.session_id, req.payload, on_event=emit))


@app.get("/api/sessions/{session_id}")
def get_session(session_id: str):
    _check_id(session_id, "session_id")
    pending, err = None, ""
    try:
        pending = _graph().pending_interrupt(session_id)
    except Exception as e:
        err = f"{type(e).__name__}: {e}"
    mem = memory.load_session(session_id)
    return {"session_id": session_id, "pending_interrupt": pending, "error": err,
            "memory": {k: mem.get(k) for k in ("order_stack", "summary", "turns", "last_intent")}}


# ---------------- 演示数据 ----------------
def enrich_order(o: dict) -> dict:
    f = data.get_flight(o["flight_no"], o["date"]) or {}
    flight = {k: v for k, v in f.items() if k != "fares"}
    return {**o, "airline_name": seed.AIRLINES.get(o["airline"], o["airline"]),
            "product_label": seed.PRODUCTS.get(o["fare_product"], (None, o["fare_product"]))[1],
            "cabin_label": CABIN_LABEL.get(o["cabin"], o["cabin"]),
            "status_label": ORDER_STATUS_LABEL.get(o["status"], o["status"]),
            "flight": flight, "flight_status": f.get("status"), "delay_min": f.get("delay_min", 0),
            "flight_status_label": FLIGHT_STATUS_LABEL.get(f.get("status"), f.get("status"))}


@app.get("/api/users")
def users():
    return {"users": [memory.public_profile(p) for p in data.list_profiles()]}


@app.get("/api/orders")
def orders(user_id: str | None = Query(default=None)):
    if user_id is not None:
        _check_id(user_id, "user_id")
    return {"orders": [enrich_order(o) for o in data.list_orders(user_id)]}


class FlightStatusReq(Body):
    status: Literal["scheduled", "delayed", "cancelled"]
    delay_min: int = Field(default=0, ge=0, le=1440)


@app.post("/api/flights/{flight_no}/{date}/status")
def flight_status(flight_no: str, date: str, req: FlightStatusReq):
    if not data.get_flight(flight_no, date):
        raise ApiError(404, "NOT_FOUND", f"航班不存在：{flight_no} {date}")
    if req.status == "delayed" and req.delay_min <= 0:
        raise ApiError(400, "VALIDATION", "延误需填写分钟数", {"delay_min": "延误分钟数必须大于 0"})
    data.set_flight_status(flight_no, date, req.status, req.delay_min)
    return {"flight": data.get_flight(flight_no, date)}


@app.post("/api/data/reset")
def data_reset():
    data.init(force=True)
    return {"ok": True}


@app.delete("/api/profiles/{user_id}/memory")
def clear_memory(user_id: str):
    _check_id(user_id, "user_id")
    if not data.get_profile(user_id):
        raise ApiError(404, "NOT_FOUND", f"用户不存在：{user_id}")
    memory.clear_profile(user_id)
    return {"user": memory.public_profile(data.get_profile(user_id))}


# ---------------- 规则库 ----------------
class RuleReq(Body):
    id: str = Field(pattern=ID)
    airline: str = Field(min_length=1, max_length=8)
    cabin: str = Field(min_length=1, max_length=8)
    fare_product: str = Field(min_length=1, max_length=32)
    scenario: str = Field(min_length=1, max_length=32)
    title: str = Field(min_length=1, max_length=200)
    content: str = Field(min_length=1, max_length=4000)


def _check_rule(r: RuleReq):
    fields = {}
    if r.airline != "*" and r.airline not in seed.AIRLINES:
        fields["airline"] = f"航司必须是 {'/'.join(seed.AIRLINES)} 或 *"
    if r.cabin not in CABIN_LABEL:
        fields["cabin"] = "舱位必须是 F/C/Y 或 *"
    if r.fare_product != "*" and r.fare_product not in seed.PRODUCTS:
        fields["fare_product"] = f"票价产品必须是 {'/'.join(seed.PRODUCTS)} 或 *"
    if r.scenario not in SCENARIOS:
        fields["scenario"] = f"场景必须是 {'/'.join(sorted(SCENARIOS))}"
    if fields:
        raise ApiError(400, "VALIDATION", "规则字段不合法", fields)


@app.get("/api/rules")
def rules(airline: str | None = None, cabin: str | None = None, fare_product: str | None = None,
          scenario: str | None = None):
    out = data.list_rules(airline or None, fare_product or None, scenario or None)
    if cabin:
        out = [r for r in out if r["cabin"] == cabin]
    return {"rules": out}


@app.post("/api/rules", status_code=201)
def create_rule(req: RuleReq):
    _check_rule(req)
    if data.get_rule(req.id):
        raise ApiError(409, "CONFLICT", f"规则 ID 已存在：{req.id}", {"id": "ID 已存在"})
    data.upsert_rule(req.model_dump())
    return {"rule": data.get_rule(req.id)}


@app.put("/api/rules/{rule_id}")
def update_rule(rule_id: str, req: RuleReq):
    _check_id(rule_id, "id")
    if req.id != rule_id:
        raise ApiError(400, "VALIDATION", "不能修改规则 ID", {"id": "与路径中的 ID 不一致"})
    _check_rule(req)
    if not data.get_rule(rule_id):
        raise ApiError(404, "NOT_FOUND", f"规则不存在：{rule_id}")
    data.upsert_rule(req.model_dump())
    return {"rule": data.get_rule(rule_id)}


@app.delete("/api/rules/{rule_id}")
def delete_rule(rule_id: str):
    _check_id(rule_id, "id")
    if not data.get_rule(rule_id):
        raise ApiError(404, "NOT_FOUND", f"规则不存在：{rule_id}")
    data.delete_rule(rule_id)
    return {"ok": True}


class SearchFilters(Body):
    airline: str | None = None
    cabin: str | None = None
    fare_product: str | None = None
    scenario: str | None = None


class SearchDebugReq(Body):
    query: str = Field(min_length=1, max_length=1000)
    filters: SearchFilters = Field(default_factory=SearchFilters)
    user_id: str | None = Field(default=None, pattern=ID)


@app.post("/api/rules/search-debug")
def search_debug(req: SearchDebugReq):
    filters = {k: v for k, v in req.filters.model_dump().items() if v}
    boost = memory.profile_boost(req.user_id) if req.user_id else {}
    t0 = time.perf_counter()
    res = retrieval.search(req.query, filters, boost)
    return {**res, "boost": boost, "ms": round((time.perf_counter() - t0) * 1000, 1)}


# ---------------- 评测 ----------------
class GoldExpect(BaseModel):
    model_config = ConfigDict(extra="allow")
    intents: list[str] = Field(default_factory=list)
    tools: list[str] = Field(default_factory=list)
    fees: dict[str, int | None] = Field(default_factory=dict)
    rule_ids: list[str] = Field(default_factory=list)
    outcome: str | None = None
    schema_errors: Any = None


class GoldCase(BaseModel):
    # 用例项允许多余字段：evaluation 模块可能逐步增加字段，页面原样保存
    model_config = ConfigDict(extra="allow")
    id: str = Field(pattern=ID)
    title: str = ""
    tags: list[str] = Field(default_factory=list)
    user_id: str = Field(pattern=ID)
    airline: str | None = None
    fare_product: str | None = None
    order: str | None = None
    question: str = Field(min_length=1, max_length=1000)
    history: list[dict[str, Any]] = Field(default_factory=list)
    clarify_answers: Any = None
    expect: GoldExpect
    forbid: list[str] = Field(default_factory=list)


class CasesReq(Body):
    cases: list[dict[str, Any]]


def _shape_errors(cases: list[dict]) -> dict:
    """本地形状校验，返回 {case_id: [{field, message}]}。"""
    errors, seen = {}, set()
    for i, c in enumerate(cases):
        cid = str(c.get("id") or f"#{i + 1}")
        try:
            GoldCase.model_validate(c)
        except ValidationError as e:
            errors.setdefault(cid, []).extend({"field": _loc(x["loc"]), "message": x["msg"]} for x in e.errors())
        if cid in seen:
            errors.setdefault(cid, []).append({"field": "id", "message": "用例 ID 重复"})
        seen.add(cid)
    return errors


def _read_cases() -> list:
    if not EVAL_CASES.exists():
        return []
    return json.loads(EVAL_CASES.read_text("utf-8"))


def _run_cli(args: list[str], timeout: int = 120) -> tuple[int, str, str]:
    """运行评测命令行（单独抽出来，方便测试替换）。"""
    p = subprocess.run([sys.executable, "-m", "air_agent.evaluation", *args], cwd=ROOT, capture_output=True,
                       text=True, timeout=timeout)
    return p.returncode, p.stdout, p.stderr


def _json_lines(text: str):
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("{"):
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


@app.get("/api/eval/cases")
def get_cases():
    return {"cases": _read_cases(), "path": "eval/gold_cases.json", "exists": EVAL_CASES.exists()}


@app.put("/api/eval/cases")
def put_cases(req: CasesReq):
    errors = _shape_errors(req.cases)
    if errors:
        raise ApiError(400, "VALIDATION", "用例格式不合法", {"cases": errors})
    EVAL_CASES.parent.mkdir(parents=True, exist_ok=True)
    if EVAL_CASES.exists():
        EVAL_CASES.with_suffix(".json.bak").write_bytes(EVAL_CASES.read_bytes())
    tmp = EVAL_CASES.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(req.cases, ensure_ascii=False, indent=2), "utf-8")
    tmp.replace(EVAL_CASES)
    return {"ok": True, "count": len(req.cases)}


@app.post("/api/eval/cases/validate")
def validate_cases(req: CasesReq):
    errors = _shape_errors(req.cases)
    if errors:
        return {"ok": False, "errors": errors, "source": "shape"}
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8") as f:
        json.dump(req.cases, f, ensure_ascii=False)
        path = f.name
    try:
        rc, out, err = _run_cli(["validate", "--cases", path])
    except subprocess.TimeoutExpired:
        raise ApiError(504, "TIMEOUT", "用例校验超时")
    finally:
        Path(path).unlink(missing_ok=True)
    for msg in _json_lines(out):
        if msg.get("type") == "validated":
            errs = msg.get("errors") or {}
            return {"ok": not errs, "errors": errs, "source": "engine"}
    raise ApiError(500, "EVAL_CLI", "评测命令行没有返回校验结果", {"stderr": err[-800:]})


def _runs_dir() -> Path:
    return S.VAR_DIR / "eval_runs"


_run_lock = threading.Lock()
_runs: dict[str, dict] = {}       # run_id -> {lines, done, rc, stderr, proc}


def _spawn(args: list[str]):
    return subprocess.Popen([sys.executable, "-m", "air_agent.evaluation", *args], cwd=ROOT, text=True,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=1)


def _active_run():
    return next((rid for rid, r in _runs.items() if not r["done"]), None)


class RunReq(Body):
    mode: Literal["offline", "deepseek"] = "offline"
    ids: list[str] | None = None


@app.post("/api/eval/runs", status_code=201)
def start_run(req: RunReq):
    if req.mode == "deepseek" and not S.get()["llm"]["api_key"]:
        raise ApiError(400, "LLM_NOT_CONFIGURED", "DeepSeek 模式需要先在配置中心填写 API Key")
    for i in req.ids or []:
        _check_id(i, "ids")
    with _run_lock:
        if (active := _active_run()):
            raise ApiError(409, "RUN_ACTIVE", f"已有评测在运行：{active}", {"run_id": active})
        run_id = f"{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:4]}"
        args = ["run", "--mode", req.mode, "--run-id", run_id]
        if req.ids:
            args += ["--ids", ",".join(req.ids)]
        state = {"lines": [], "done": False, "rc": None, "stderr": "", "mode": req.mode,
                 "started": time.time(), "total": len(req.ids) if req.ids else None}
        _runs[run_id] = state
        try:
            proc = _spawn(args)
        except Exception as e:
            state.update(done=True, rc=-1, stderr=str(e))
            raise ApiError(500, "EVAL_CLI", f"无法启动评测：{e}")

    def pump():
        try:
            for line in proc.stdout:
                line = line.strip()
                if line.startswith("{"):
                    try:
                        state["lines"].append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
            state["stderr"] = proc.stderr.read()[-2000:]
            state["rc"] = proc.wait()
        finally:
            state["done"] = True
    threading.Thread(target=pump, daemon=True, name=f"eval-{run_id}").start()
    return {"run_id": run_id}


@app.get("/api/eval/runs/{run_id}/stream")
async def run_stream(run_id: str):
    _check_id(run_id, "run_id")
    state = _runs.get(run_id)
    if state is None:
        if (_runs_dir() / f"{run_id}.json").exists():
            run = json.loads((_runs_dir() / f"{run_id}.json").read_text("utf-8"))
            async def done_only():
                yield sse("done", {"run_id": run_id, "summary": run.get("summary")})
            return StreamingResponse(done_only(), media_type="text/event-stream", headers=SSE_HEADERS)
        raise ApiError(404, "NOT_FOUND", f"评测运行不存在：{run_id}")

    async def gen():
        i, saw_done = 0, False
        while True:
            lines = state["lines"]
            while i < len(lines):
                msg = lines[i]
                i += 1
                t = msg.get("type", "log")
                saw_done = saw_done or t == "done"
                yield sse(t if t in ("progress", "done") else "log", msg)
            if state["done"] and i >= len(state["lines"]):
                if not saw_done:
                    yield sse("error", {"code": "EVAL_CLI", "message": f"评测进程退出（rc={state['rc']}）",
                                        "stderr": state["stderr"][-800:]})
                break
            await asyncio.sleep(0.2)
    return StreamingResponse(gen(), media_type="text/event-stream", headers=SSE_HEADERS)


@app.get("/api/eval/runs")
def list_runs():
    out = []
    d = _runs_dir()
    for p in sorted(d.glob("*.json"), reverse=True) if d.exists() else []:
        try:
            r = json.loads(p.read_text("utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        out.append({"id": r.get("id", p.stem), "created_at": r.get("created_at"), "mode": r.get("mode"),
                    "summary": r.get("summary")})
    out.sort(key=lambda r: r.get("created_at") or "", reverse=True)
    return {"runs": out, "active": _active_run()}


@app.get("/api/eval/runs/{run_id}")
def get_run(run_id: str):
    _check_id(run_id, "run_id")
    p = _runs_dir() / f"{run_id}.json"
    if not p.exists():
        raise ApiError(404, "NOT_FOUND", f"评测运行不存在：{run_id}")
    return json.loads(p.read_text("utf-8"))


# ---------------- 配置 ----------------
# (最小值, 最大值)；路径用点号。budget_share 另行校验总和。
RANGES = {
    "llm.timeout": (1, 120), "query.clarify_threshold": (0, 1), "query.step_back_min_hits": (0, 10),
    "planner.max_rounds": (1, 20), "planner.max_tool_retries": (0, 5),
    "retrieval.k_recall": (1, 200), "retrieval.k_fuse": (1, 200), "retrieval.k_rerank": (1, 100),
    "retrieval.k_mmr": (1, 50), "retrieval.k_inject": (1, 20), "retrieval.rrf_k": (1, 200),
    "retrieval.rerank_min": (0, 1), "retrieval.mmr_lambda": (0, 1), "retrieval.profile_boost": (0, 1),
    "retrieval.dense_min.hybrid": (0, 1), "retrieval.dense_min.fallback": (0, 1),
    "retrieval.dense_strict.hybrid": (0, 1), "retrieval.dense_strict.fallback": (0, 1),
    "memory.recent_turns": (1, 20), "memory.order_stack": (1, 20), "memory.token_budget": (200, 32000),
    "verify.threshold": (0, 1), "verify.max_retries": (0, 5),
}
INT_FIELDS = {"llm.timeout", "query.step_back_min_hits", "planner.max_rounds", "planner.max_tool_retries",
              "retrieval.k_recall", "retrieval.k_fuse", "retrieval.k_rerank", "retrieval.k_mmr", "retrieval.k_inject",
              "retrieval.rrf_k", "memory.recent_turns", "memory.order_stack", "memory.token_budget",
              "verify.max_retries"}


class ConfigPatch(Body):
    mode: Literal["offline", "deepseek"] | None = None
    llm: dict[str, Any] | None = None
    query: dict[str, Any] | None = None
    planner: dict[str, Any] | None = None
    retrieval: dict[str, Any] | None = None
    memory: dict[str, Any] | None = None
    verify: dict[str, Any] | None = None


def _get_path(cfg, path):
    for k in path.split("."):
        cfg = cfg[k]
    return cfg


def _merge_preview(base, patch):
    out = copy.deepcopy(base)
    for k, v in patch.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict) and k != "budget_share":
            out[k] = _merge_preview(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def _range_errors(cfg) -> dict:
    fields = {}
    for path, (lo, hi) in RANGES.items():
        try:
            v = _get_path(cfg, path)
        except (KeyError, TypeError):
            continue
        if not isinstance(v, (int, float)) or isinstance(v, bool):
            continue
        if path in INT_FIELDS and v != int(v):
            fields[path] = "必须是整数"
        elif not lo <= v <= hi:
            fields[path] = f"取值范围 {lo}–{hi}"
    r = cfg.get("retrieval", {})
    chain = ["k_recall", "k_fuse", "k_rerank", "k_mmr", "k_inject"]
    for a, b in zip(chain, chain[1:]):
        if all(isinstance(r.get(x), (int, float)) for x in (a, b)) and r[a] < r[b]:
            fields.setdefault(f"retrieval.{b}", f"不能大于 {a}（漏斗逐层收窄）")
    share = (cfg.get("memory") or {}).get("budget_share")
    if isinstance(share, dict):
        need = set(memory.PRIORITY)
        if set(share) != need:
            fields["memory.budget_share"] = f"必须且只能包含 {'/'.join(memory.PRIORITY)}"
        elif any(not isinstance(v, (int, float)) or v < 0 or v > 1 for v in share.values()):
            fields["memory.budget_share"] = "每项取值 0–1"
        elif sum(share.values()) > 1.0001:
            fields["memory.budget_share"] = f"各项之和不能超过 1（当前 {sum(share.values()):.2f}）"
    base = (cfg.get("llm") or {}).get("base_url")
    if isinstance(base, str) and not re.match(r"^https?://", base):
        fields["llm.base_url"] = "必须以 http:// 或 https:// 开头"
    return fields


@app.get("/api/config")
def get_config():
    return S.public()


@app.put("/api/config")
def put_config(req: ConfigPatch):
    patch = req.model_dump(exclude_none=True)
    if "llm" in patch and "****" in str(patch["llm"].get("api_key", "")):
        patch["llm"].pop("api_key")        # 打码值原样回传 = 不修改
    if patch.get("llm") == {}:
        patch.pop("llm")
    if not patch:
        return S.public()
    fields = _range_errors(_merge_preview(S.get(), patch))
    if fields:
        raise ApiError(400, "VALIDATION", "配置校验失败", fields)
    try:
        S.update(patch)
    except (KeyError, TypeError, ValueError) as e:
        msg = str(e).strip("'\"")
        m = re.search(r"([a-z_]+(?:\.[a-z_]+)+)", msg)
        raise ApiError(400, "VALIDATION", msg, {m.group(1) if m else "config": msg})
    return S.public()


@app.post("/api/config/reset")
def reset_config():
    key = S.get()["llm"]["api_key"]
    S.reset()
    if key:
        S.update({"llm": {"api_key": key}})
    return S.public()


@app.post("/api/config/test-llm")
def test_llm():
    cfg = S.get()["llm"]
    key = cfg["api_key"]
    if not key:
        return {"ok": False, "error": "未配置 API Key"}
    t0 = time.perf_counter()
    try:
        r = httpx.post(f"{cfg['base_url'].rstrip('/')}/chat/completions", timeout=min(cfg["timeout"], 15),
                       headers={"Authorization": f"Bearer {key}"},
                       json={"model": cfg["model"], "messages": [{"role": "user", "content": "ping"}],
                             "max_tokens": 1, "temperature": 0})
        ms = round((time.perf_counter() - t0) * 1000)
        if r.status_code != 200:
            return {"ok": False, "ms": ms, "error": f"HTTP {r.status_code}：{r.text[:200].replace(key, '****')}"}
        return {"ok": True, "ms": ms, "model": r.json().get("model", cfg["model"])}
    except Exception as e:
        return {"ok": False, "error": f"{type(e).__name__}: {str(e).replace(key, '****')[:200]}"}


@app.get("/api/status")
def status():
    cfg = S.get()
    return {"mode": cfg["mode"], "retrieval": retrieval.status(), "deepseek_configured": bool(cfg["llm"]["api_key"]),
            "model": cfg["llm"]["model"], "now": S.now().isoformat(), "demo": True,
            "eval_active": _active_run()}


# 静态前端（放在所有 /api 路由之后）
if WEB_DIR.exists():
    app.mount("/", StaticFiles(directory=WEB_DIR, html=True), name="web")


def main():
    ap = argparse.ArgumentParser(description="退改签助手 Web 服务")
    ap.add_argument("--port", type=int, default=8770)
    args = ap.parse_args()
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=args.port, log_level="info")


if __name__ == "__main__":
    main()
