# 机票退改签 Agent（MVP）

基于 LangGraph 的机票改签/退票智能助手：理解乘客诉求 → 检索退改规则 → 运价引擎算钱 → 生成方案 → 校验防幻觉 → 乘客确认后办理。

> 所有航司、航班、订单、用户与规则均为**虚构数据**，由 `air_agent/seed.py` 按「今天」确定性生成，仅用于演示。

## 快速启动

```bash
./start.sh                      # 安装依赖（uv sync）并启动
```

- Web 工作台（`app.py`，http://127.0.0.1:8770）正在接入中。`app.py` 存在时 `./start.sh` 会启动它，参数原样传给 `app.py`。
- 没有 `app.py` 时，`./start.sh` 进入命令行对话：

```bash
uv run python -m air_agent --user u_zhang --order "ORD$(date +%Y%m%d)001"
```

命令行里输入问题即可，会逐行打印节点 trace；遇到追问直接输入答案，遇到办理确认输入 `确认`/`y` 或 `取消`/`n`，空行退出。

演示用户：`u_zhang`（张三）、`u_li`（李四）、`u_wang`（王五），订单号为 `ORD<今天YYYYMMDD><序号 001-012>`。如果种子数据是在别的日期生成的，先重建：

```bash
uv run python -c "from air_agent import data; data.init(force=True)"
```

## 接入 DeepSeek

默认是规则模式（`offline`），不需要任何 Key。切换到 DeepSeek：

```bash
cp config.example.json config.json
# 编辑 config.json：填写 llm.api_key，并把 mode 改为 "deepseek"
```

```json
{
  "mode": "deepseek",
  "llm": {"base_url": "https://api.deepseek.com", "api_key": "sk-...", "model": "deepseek-chat", "timeout": 30}
}
```

`config.json` 已在 `.gitignore` 中，不会被提交。DeepSeek 模式下 LLM 负责 ReAct 决策、意图拆解、方案措辞、摘要和事实核查；**金额始终来自运价引擎**。LLM 调用失败时自动回退到规则计划表（trace 中 `source=fallback`）。也可以单次指定：`uv run python -m air_agent --mode deepseek`。

## 五项工作与代码对照

| 工作项 | 图节点 | 代码位置 |
|---|---|---|
| 一 ReAct 规划编排 | agent_node / schema_validate / checkpoint_gate / tool_exec / retry_node | `air_agent/graph.py`、`planner.py`、`tools.py` |
| 二 Query 理解 | order_align / query_understand / clarify / step_back | `air_agent/query.py` |
| 三 混合检索与精排 | tool_exec(tool_rule_retrieve) | `air_agent/retrieval.py` |
| 四 上下文与记忆 | memory_node / finish | `air_agent/memory.py` |
| 五 方案生成与质量保障 | tool_scheme_gen / verify_node / human_review / await_confirm / tool_issue_ticket | `air_agent/scheme.py`、`verify.py`、`evaluation.py` |

其他模块：`engine.py` 运价引擎（金额的唯一来源）、`data.py` SQLite 数据层、`seed.py` 虚构数据、`llm.py` DeepSeek 客户端与测试用 `FakeLLM`、`settings.py` 配置。

主流程：

```
user_input → memory_node → order_align → query_understand →（clarify）→ agent_node ⇄ schema_validate
  → checkpoint_gate → tool_exec →（step_back）→ … → sub_done → tool_scheme_gen → verify_node
  →（retry_node）→ finish | human_review → finish →（await_confirm → tool_issue_ticket）
```

- 追问（clarify）和办理确认（await_confirm）用 LangGraph `interrupt()` 实现，检查点存在 `var/checkpoints.db`，服务重启后可以继续 `resume`。
- 防跳步：`checkpoint_gate` 要求算钱前必须先检索规则；被打回超过 2 次转人工。
- 工具参数错误（如相对日期「后天」）会把结构化错误反馈给下一轮 ReAct；无法修正时追问乘客或转人工。
- 方案的依据规则必须被检索到并被引用（`scheme.ensure_fee_rule` 会把运价引擎用到的规则补进结果），`verify` 再逐项比对金额、核查论断、检查引用。
- 同一会话的整轮对话持有 `memory.session_lock`，并发请求不会互相覆盖记忆。
- Python 接口：`graph.run_turn(session_id, user_id, message, order_id=None, mode=None, client=None, on_event=None)`、`graph.resume(session_id, payload, ...)`、`graph.pending_interrupt(session_id)`。

## 演示脚本

以下以 `D=$(date +%Y%m%d)` 为订单号前缀，在命令行或 Web 工作台输入均可。

| # | 场景 | 用户 / 订单 | 输入 | 预期 |
|---|---|---|---|---|
| 1 | 改签差价 | u_zhang / `ORD${D}001` | 帮我改到大后天 | 改期费 + 票面差价 + 合计，等待确认；输入「确认」后出新票号 |
| 2 | 改签加退票 | u_zhang / `ORD${D}001` | 改到大后天，顺便看看退票多少钱 | 两个子诉求、两个方案（CHANGE、REFUND） |
| 3 | 航班取消 | u_wang / `ORD${D}007` | 航班取消了，帮我退票 | 非自愿退票，退票费 0 元，全额退款，引用 GEN-CANCEL |
| 4 | 指代追问 | u_zhang / 先做 #1，再输入 | 那退票呢 | 从订单栈取到同一订单，给出退票方案 |
| 5 | 模糊日期触发 Schema 校验 | u_zhang / `ORD${D}001` | 改到下个月初 | schema_validate 报 TYPE 错误 → 追问日期 → 选定后给出方案 |
| 6 | 无关问题转人工 | u_wang / 不选订单 | 宠物能带上飞机吗 | 先追问意图；选「政策咨询」后检索不到依据 → retry_node → 转人工 |

另外：u_zhang 不选订单输入「我要退票」，会先追问订单（选 `ORD${D}003`）。

## 测试与评测

```bash
uv run pytest -q                 # 单元测试（默认用 TF-IDF + BM25 降级检索，不下载模型）
uv run pytest -m slow -v         # 真实 BGE-small-zh + FlashRank 检索（需要下载模型）
```

金标准评测（固定在 2026-09-26 10:00 运行，使用独立沙箱库 `var/eval_sandbox/`，每条用例前重置数据，不影响演示数据）：

```bash
uv run python -m air_agent.evaluation build              # 生成 eval/gold_cases.json（65 条）
uv run python -m air_agent.evaluation validate           # 校验用例 → {"type":"validated","errors":{}}
uv run python -m air_agent.evaluation run --mode offline # 运行，结果写入 var/eval_runs/<run_id>.json
uv run python -m air_agent.evaluation run --mode offline --ids o01-chg,policy-01 --run-id demo
```

**期望金额的来源**：用例里的期望金额由构建脚本调用运价引擎生成，`validate` 时再用引擎重算比对。所以评测检验的是「Agent 是否调用了正确的工具、拿到了正确的引擎结果，并原样写进方案」，而不是检验运价引擎本身的计费规则（引擎由 `tests/test_engine.py` 单独覆盖）。

### 实测指标（规则模式 offline，检索后端 fallback，2026-11-05）

| 指标 | 数值 |
|---|---|
| 用例数 n | 65 |
| 工具调用准确率 tool_acc | 1.00 |
| 政策命中率 policy_hit | 0.98 |
| 正确率 correctness | 1.00 |
| 转人工率 human_rate | 0.046（3 条转人工用例） |
| 平均延迟 / P95 | 29.1 ms / 37.3 ms |
| 平均上下文 tokens | 266 |

唯一失败原因是 `rule_miss` 1 条（policy-01「东方云航特价票的退订规定是什么」：降级检索把特价票的改签规则排在了退票规则前面）。`rule_miss` 与 `wrong_tools` 只计入政策命中率与工具准确率，不计入正确率。DeepSeek 模式的评测需要配置 Key 后运行 `--mode deepseek`，尚未实测。

用例覆盖：12 张演示订单 × 4 种问法（改签 / 退票 / 改签+退票 / 航班状态），8 条政策咨询（四家航司），追问（订单、日期、模糊日期）、指代（回程、省略、订单栈）各 3 条，3 条转人工。
