# 机票退改签 Agent MVP 设计文档

- 日期：2026-11-04
- 来源：cocochimp/Agent-Tutorial `web/projects/air-ticket/work-items.json`（五项工作内容）
- 目标：做一个可以本地运行、可以交互演示的 MVP，让学员能把简历上的五项工作逐项跑给面试官看，并能讲清楚实现。

## 0. 范围与约束

**要做的**
- 独立项目，路径 `/Users/Doc/agent/air-ticket-agent`，不依赖 Agent-Tutorial 的代码。
- 用 LangGraph 编排，两种决策模式都要：
  - **规则模式**：不需要 API Key，可以离线运行，结果确定，测试默认用它；
  - **DeepSeek 模式**：兼容 OpenAI 接口，真实调用 Function Calling。
- 前端有 5 个页面：对话工作台、评测中心、规则库、订单与用户、配置中心。
- 金标准评测集可以在页面上配置、校验、运行和对比。

**不做的**
- 真实的 GDS、支付、出票接口，全部用虚构数据和本地运价引擎代替；
- 多租户、登录和权限（页面顶部切换演示用户即可）；
- 移动端专门适配（只保证宽度小于 1024 时不错乱）；
- LoRA 微调、Docker 部署。

**数据声明**：航司、规则、票价、用户全部虚构。航司代码沿用教学材料里的写法（如 MU），页面统一标注「演示数据」。

## 1. 技术栈与目录

- Python 3.13，用 uv 管理依赖。
- 后端：LangGraph ≥1.0、langgraph-checkpoint-sqlite、FastAPI、Uvicorn、Pydantic v2、httpx。
- 检索：fastembed（BAAI/bge-small-zh-v1.5，ONNX，CPU 运行）、qdrant-client（本地模式）、rank-bm25 + jieba（中文分词）、flashrank（Cross-Encoder 精排）。
- 存储：SQLite，用于业务数据、会话记忆、用户画像、评测记录和 LangGraph 检查点。
- 前端：原生 HTML + CSS + ES Modules，不需要构建；图标用内联的 Lucide SVG。
- 测试：pytest。

```
air-ticket-agent/
  pyproject.toml  start.sh  config.example.json  README.md
  air_agent/
    settings.py     运行配置：读取 config.json 与默认值，支持运行时修改
    state.py        AgentState（TypedDict）、ParsedQuery 等 Pydantic 模型
    data.py         初始化 SQLite、加载种子数据，提供 orders/flights/profiles 的读写
    engine.py       运价引擎：计算改签费、差价、退票费（唯一的金额来源）
    tools.py        5 个工具及其参数 Schema，统一执行入口 run_tool()
    query.py        工作二：订单字段对齐、意图与多诉求拆解、指代消解、日期换算、HyDE、Step-back
    retrieval.py    工作三：向量 + BM25 → RRF → 硬过滤 → 精排 → MMR，每层留痕
    memory.py       工作四：短期会话记忆、长期用户画像、Token 预算分配
    verify.py       工作五：金额比对、论断事实核查、置信度
    llm.py          DeepSeek 客户端，以及测试用的假客户端
    graph.py        StateGraph 定义、条件边、interrupt、检查点
    evaluation.py   用例校验、批量运行评测、计算指标
    trace.py        节点 trace 事件结构，供 SSE 推送
  app.py            FastAPI 路由与 SSE
  web/              index.html、css/、js/（每个页面一个模块）
  data/seed/        orders.json、flights.json、profiles.json、rules.json
  eval/             gold_cases.json（初始约 60 条）
  tests/
  docs/superpowers/specs/
```

## 2. 状态图（工作一：ReAct 规划编排）

```
START → user_input → memory_node → order_align → query_understand
  query_understand ─(缺关键字段 / 置信度 < 0.6)→ clarify（interrupt，等用户选择或补充）→ query_understand
  query_understand ─(需要泛化)→ step_back → agent_node
  query_understand ─(其他)→ agent_node

agent_node（ReAct：给出 Thought，再选 Action）
  ─ 选了工具 → schema_validate
        ├ 通过 → checkpoint_gate_pre（调 recalc_fee 前必须已调 rule_retrieve，否则打回 agent_node 并附原因）
        │        → tool_exec（执行 flight_status / rule_retrieve / recalc_fee）→ agent_node
        └ 不通过 → error_feedback 写回状态 → agent_node（同一工具最多重试 1 次，超过就进 human_review）
  ─ 信息够了 → tool_scheme_gen → verify_node
  ─ 已到 5 轮上限 → human_review

verify_node
  ├ confidence ≥ 阈值（默认 0.85）→ finish
  ├ 不到阈值且 retries < 上限（默认 1）→ retry_node（回灌具体错误）→ tool_scheme_gen
  └ 仍不通过 → human_review（结束本轮，outcome=HUMAN，页面显示「已转人工」并列出未通过的校验项）

finish：合并多个子诉求的结果，写入短期记忆和长期画像；有可办理的方案时进入 await_confirm，否则结束（END）

await_confirm（interrupt，返回可办理方案和金额明细）
  ├ 用户点「确认改签/退票」→ /api/resume → tool_issue_ticket（引擎重算金额，按 confirm_id 幂等）→ END
  ├ 用户点「取消」→ END（outcome=CANCELLED）
  └ 用户直接发新消息 → 新一轮从 START 开始，丢弃未确认的方案
```

- **多诉求**：`query_understand` 产出 `sub_queries`，`agent_node` 一次处理一个子诉求，处理完写入 `results[i]` 后切到下一个。所有子诉求完成后才进入 `tool_scheme_gen`。
- **两种模式**：
  - 规则模式下，`agent_node` 按意图查固定计划表：改签是 flight_status → rule_retrieve → recalc_fee；退票是 rule_retrieve → recalc_fee；纯政策咨询只有 rule_retrieve。
  - DeepSeek 模式下，把工具 Schema 交给模型选择。
  - 两种模式输出同一种结构 `{thought, action, args}`。
- **检查点**：用 SqliteSaver 保存，`thread_id = session_id`。服务重启后，处于 interrupt 状态的会话可以继续执行。
- **trace**：每个节点产出 `{node, status, started_at, ms, input, output, note}`，通过 SSE 实时推给前端。

### 工具与参数 Schema

| 工具 | 入参 | 出参 |
|---|---|---|
| `tool_flight_status` | flight_no（`^[A-Z0-9]{2}\d{3,4}$`）、date（YYYY-MM-DD） | status: scheduled/delayed/cancelled，delay_min |
| `tool_rule_retrieve` | query、airline、cabin、fare_product、scenario | 前 3 条规则及漏斗 trace |
| `tool_recalc_fee` | order_id（`^ORD\d{11}$`）、action（CHANGE/REFUND）、target_date（改签时必填）、target_flight（可选） | change_fee、fare_diff、refund_fee、refund_amount、currency、rule_id |
| `tool_scheme_gen` | sub_results、rules、fees、feedback | SchemeOutput（见第 6 节） |
| `tool_issue_ticket` | order_id、action、confirm_id | ticket_no、status |

- **校验失败时**：Pydantic 的 ValidationError 被转成 `{field, expected, got}` 列表写回状态，例如「target_date 必须为 YYYY-MM-DD 绝对日期」。
- **金额来源**：所有金额只能来自 `engine.py`。运价引擎按「规则表 + 起飞前剩余时间 + 航班状态」计算，例如航班取消时免费改签、全额退款。

## 3. Query 理解（工作二）

输出 `ParsedQuery`：

```
intents: [CHANGE | REFUND | POLICY | STATUS]
sub_queries: [{intent, text, target_date?}]
order_id, airline, cabin, fare_product, scenario
missing: [字段名]
rewritten: 改写后的独立问题
step_back: 泛化后的问题（可为空）
confidence: 0~1
```

- **订单字段对齐**：用户在工作台选了订单，或从记忆里解析出订单时，用订单数据补全 airline、cabin、fare_product。用户的口语说法通过同义词表映射，比如「打折票 / 特价票」映射为 `Y_DISCOUNT`，「头等」映射为 `F`。
- **多诉求拆解**：规则模式按连接词（「顺便 / 另外 / 还有 / 再」）和意图关键词切分；DeepSeek 模式让模型输出 JSON，再用 Pydantic 校验。
- **指代消解**：「它 / 这张 / 上一张 / 回程」按会话订单栈解析，规则依次是最近提到的订单、最近操作的订单、按航段方向匹配。解析不出唯一订单时进入 clarify，页面上给出订单选项让用户点选。
- **日期换算**：以服务器当天为基准（测试里可以注入固定日期），换算「今天 / 明天 / 后天 / 大后天 / 下周X / X月X号」。
- **Step-back**：问题里有具体舱位代码或票价产品，而第一轮检索的精排结果不到 2 条时，把问题泛化成「{舱位大类}{场景}规则」，再检索一次。
- **HyDE**：第一轮召回为空，或精排后没有结果时，生成一段假想的规则条文去检索（规则模式用模板，DeepSeek 模式由模型生成）。
- **置信度**：规则模式下，缺 1 个关键字段（订单号或意图）扣 0.3，缺 1 个可选字段（例如改签缺日期）扣 0.2。

## 4. 混合检索与精排（工作三）

- **规则库**：约 50 条，覆盖 4 家虚构航司 × 舱位或票价产品（F / C / Y / Y_DISCOUNT / 特价 / 团队）× 场景（自愿改签、自愿退票、航班延误、航班取消、误机）。每条规则包含 `id, airline, cabin, fare_product, scenario, title, content, fee_table`。
- **漏斗**（所有参数都能在配置中心调整，括号里是默认值）：
  1. 向量召回 Top-K1（60）：bge-small-zh 编码，存在 Qdrant 本地集合里；
  2. BM25 召回 Top-K1（60）：用 jieba 分词（中文）加航司和舱位代码这类原词；
  3. RRF 融合（k=60），去重后取 K2（40）；
  4. 硬过滤：只保留 airline 相同（或通用规则）、cabin 匹配的规则；
  5. Cross-Encoder 精排，取 K3（10），分数低于阈值（0.6，按归一化分数）的截掉；
  6. MMR 去冗余（λ=0.7），取 K4（5）；
  7. 前 K5（3）条注入生成。
- **个性化 boost**：用户画像里偏好的航司和舱位，在精排分数上乘以 `1 + boost`（默认 0.1）。
- **留痕**：每层记录条数、规则 ID 和得分，写入 trace，供前端画漏斗和检索调试器使用。
- **降级**：BGE 或 FlashRank 模型加载失败时（首次运行要下载，约 100MB），向量召回改用字符 n-gram TF-IDF，精排改用 BM25 分数加字段匹配加权。trace 和页面顶部都会标出「降级检索」。
- **重建索引**：规则库有增删改时，增量更新 BM25 和向量集合。

## 5. 上下文与记忆（工作四）

- **短期记忆**（按 `session_id`，存在 SQLite 表 `session_memory`）：
  - `order_stack`：最近提到或操作过的订单，最多 5 个；
  - `progress`：每个子诉求进行到哪一步；
  - `recent_turns`：最近 N 轮原文（默认 4）；
  - `summary`：更早轮次的滚动摘要（规则模式抽取要点，DeepSeek 模式由模型总结）。
- **长期画像**（按 `user_id`，存在表 `user_profile`）：常走航线及次数、偏好航司和舱位、退改偏好（倾向改签或退票）、价格敏感度（低 / 中 / 高，按历史上选更便宜方案的比例算）。`finish` 节点写入画像，相同的事实只更新计数，不重复追加。
- **Token 预算**：总预算默认 3000，用 `len(text) / 1.5` 粗略估算 token 数。按以下顺序分配，超出时从优先级低的开始压缩或截断：
  1. 系统规则（固定）；
  2. 当前订单（必带）；
  3. 前 3 条命中规则；
  4. 会话摘要和最近几轮原文；
  5. 用户画像。
- **trace 记录**：每块的 token 占用，以及「不做压缩、全量拼接」时的 token 数，前端据此展示节省比例。

## 6. 方案生成与质量保障（工作五）

`SchemeOutput`：

```
options: [{type: CHANGE|REFUND, title, fee_items: [{name, amount}], total, rule_refs: [rule_id], notes}]
text: 给乘客看的说明文字
```

- **生成**：规则模式用模板生成。DeepSeek 模式要求模型输出 JSON，并且 Prompt 规定金额只能引用工具返回的字段名，例如 `{change_fee}`。模型写出数字后，系统用工具的实际值回填或覆盖。
- **`verify_node`**：
  1. **金额校验**：用正则抽出 text 和 options 里的全部金额，逐项和 `tool_recalc_fee` 的结果比对。有不一致就记 `quote_mismatch`，同时给出期望值。
  2. **事实核查**：把 text 按句拆成论断。每句里涉及退改条件的关键词（时间点、是否可改、手续费比例），必须能在 `rule_refs` 引用的规则原文里找到依据，否则记 `unsupported_claim`。规则模式用关键词和数字匹配；DeepSeek 模式再加一次 Fact-Checker 调用，给每条论断打 0~1 分。
  3. **引用合法性**：`rule_refs` 必须在本次检索结果里。
  4. **置信度**：`confidence = min(金额是否一致, 有依据的论断比例, 引用是否合法)`。
- **重试**：`retry_node` 把具体错误（例如「改期费应为 580，你写了 600」「删除无依据的『免费改签』」）作为 feedback 回灌给生成步骤。重试次数用完还不通过，进入 `human_review`。
- **出票**：必须经过 `await_confirm`。`confirm_id` 保证重复点击不会重复出票。

## 7. 评测（金标准）

用例结构：

```
id, title, tags[], airline, cabin, user_id, order_id, question, history[],
expect: {intents[], tools[] (按顺序), fees{...}, rule_ids[] (必须包含), outcome: ANSWER|CLARIFY|HUMAN},
forbid: [quote_mismatch | unsupported_claim | skip_rule | wrong_order ...]
```

- **初始用例**：约 60 条，覆盖 4 家航司、各类舱位、单诉求和多诉求、指代追问、航班取消、缺参追问、应当转人工的情况。
- **用例校验**（保存前执行）：
  - 订单和用户存在；
  - 规则 ID 存在；
  - 用运价引擎重算 expect.fees，结果必须一致（不一致时提示「期望金额与引擎不符，引擎值为 X」）；
  - 工具名合法；
  - 不允许重复 ID。
- **指标**：
  - **工具调用准确率**：实际工具序列与期望序列一致，且每次调用都通过 Schema 校验的比例；
  - **政策命中率**：expect.rule_ids 全部出现在前 3 条结果里的比例；
  - **方案正确率**：outcome 符合，金额全部一致，且没有出现 forbid 里列出的错误的比例；
  - 另外统计转人工率、平均和 P95 延迟、平均 Token、失败原因分布，以及按航司 × 舱位分组的正确率。
- **运行记录**：每次运行保存配置快照、模式和逐条结果（含 trace），可以对比两次运行。
- **命令行**：`uv run python -m air_agent.evaluation --mode offline`，输出与页面一致的指标。

## 8. 后端 API

| 方法 | 路径 | 说明 |
|---|---|---|
| POST | `/api/chat` | `{session_id, user_id, order_id?, message}`，返回 SSE：`trace` / `token` / `interrupt` / `final` / `error` 事件 |
| POST | `/api/resume` | `{session_id, payload}`，用于追问回答、出票确认或取消、人工结论，返回 SSE |
| GET | `/api/sessions/{id}` | 当前状态、记忆、最近一次 trace |
| GET/POST/PUT/DELETE | `/api/orders`、`/api/flights/{no}/status`、`/api/profiles` | 演示数据管理 |
| GET/POST/PUT/DELETE | `/api/rules`，POST `/api/rules/search-debug` | 规则库管理与检索调试 |
| GET/POST/PUT/DELETE | `/api/eval/cases`，POST `/api/eval/cases/validate`、`/import`，GET `/export` | 用例管理 |
| POST | `/api/eval/runs` | 创建一次评测运行；GET `/api/eval/runs/{id}/stream` 推送进度（SSE）；GET `/api/eval/runs`、`/{id}` 查询结果 |
| GET/PUT | `/api/config`，POST `/api/config/reset` | 运行参数；DeepSeek Key 只写不回显（返回打码值） |

- 所有请求体都用 Pydantic 校验（`extra="forbid"`），错误统一返回 `{error: {code, message, fields}}`。
- 服务只监听 127.0.0.1。

## 9. 前端

**视觉风格**：参考 UI UX Pro Max 的 AI-Native UI 风格。
- **配色**：默认深色，可切换浅色。背景 `#0B0F1A`，卡片 `#111827`，边框 `#1F2937`，主色 `#6366F1`，成功 `#10B981`，警告 `#F59E0B`，危险 `#EF4444`。浅色主题另配一套语义 token。
- **字体**：系统中文字体（PingFang SC / Microsoft YaHei）；金额、Token、得分用等宽数字。
- **其他**：圆角 12px，细边框加柔和投影；图标用 Lucide SVG，不用 emoji；动效 150–250ms，遵守系统的「减少动态效果」设置。
- **导航**：左侧窄导航栏，5 个页面，用 hash 路由（`#/chat`、`#/eval`、`#/rules`、`#/data`、`#/config`），每个页面可以直接打开。

**页面**
1. **对话工作台**
   - 左侧（约 55%）：订单卡片条、对话流（流式阶段提示、方案卡片、追问选项、转人工卡片）、6 条快捷示例、输入框。
   - 右侧（约 45%），4 个标签页：
     - 流程：节点时间线，可展开看入参和出参 JSON；
     - 检索：漏斗图，每层可展开看规则和得分；
     - 记忆：订单栈、摘要、画像，以及 Token 预算条形图（与全量拼接对比）；
     - 校验：金额比对表、论断依据、置信度仪表。
   - 联动：点方案卡上的规则引用，右侧切到检索页并高亮对应规则。
   - 出票确认：用模态框展示金额明细，支持键盘操作和 Esc 关闭，确认后调用 `/api/resume`。
2. **评测中心**：用例表格（筛选、搜索）、用例编辑抽屉（表单分组、字段旁边显示校验错误）、导入导出、运行面板（选范围和模式，显示进度条）、结果看板（指标卡、航司 × 舱位热力图、失败原因分布条形图）、失败用例下钻（期望和实际左右对比，可打开 trace）、历史运行对比。
3. **规则库**：规则表（按航司、舱位、场景筛选）、编辑抽屉、检索调试器（输入问题，查看漏斗每层结果）。
4. **订单与用户**：订单表；航班状态切换（正常 / 延误 / 取消）；用户画像卡片，可以清空长期记忆。
5. **配置中心**：参数按模块分组（决策模式与 DeepSeek、Query 理解、检索漏斗、记忆预算、质量保障），每项给出说明和取值范围，可以恢复默认。DeepSeek 配置提供「测试连接」按钮。

**可访问性**：正文 16px，对比度 ≥ 4.5:1，焦点框可见，状态同时用图标和文字表示，图表配图例和数值标签，可点击区域 ≥ 40px。

**宽度适配**：主要针对 1280–1920 宽。小于 1024 时，右侧执行面板收成抽屉。

## 10. 错误处理

- **DeepSeek 调用出错**（超时、HTTP 错误、JSON 解析失败）：当前步骤自动降级到规则模式，在 trace 里标 `fallback`，页面提示「本步已降级为规则模式」。
- **未配置 Key 时选了 DeepSeek 模式**：配置中心和顶栏提示，请求直接返回清晰的错误，不静默降级。
- **工具或引擎异常**：转成结构化错误写进 trace，进入 `human_review`，不在页面上抛出堆栈。
- **检索模型加载失败**：按第 4 节降级。
- **SSE 断开**：前端可以重新拉取 `/api/sessions/{id}`，恢复当前状态（包括 interrupt）。
- **输入限制**：消息长度 ≤ 1000 字；`session_id` 和 `user_id` 只允许 `[\w-]{1,64}`。

## 11. 测试

- **单元测试**：运价引擎各场景的金额、工具 Schema（四类错误）、日期换算、多诉求拆解、指代消解、RRF/MMR 计算、硬过滤、Token 预算裁剪、金额校验与事实核查、用例校验。
- **流程测试**（规则模式）：单诉求改签、改签加退票双诉求、缺日期追问后恢复、跳过检索直接算费被打回、置信度不足重试后转人工、出票确认（含幂等）、服务重启后从检查点恢复。
- **DeepSeek 模式**：用假客户端回放预设的工具调用，包括第一次参数错误、第二次修正，验证 error_feedback 回灌和降级逻辑。
- **API 测试**：用 FastAPI TestClient 测 SSE 事件顺序、配置修改实时生效、规则编辑后能检索到、评测运行。
- **评测**：规则模式下跑完整金标准用例，方案正确率作为回归基线写进测试（≥ 90%）。
- **页面验收**：用 ego-lite 把 5 个页面的主要流程走一遍并截图，深色和浅色主题各检查一次。

## 12. 交付与运行

- `./start.sh`：执行 `uv sync`，初始化数据库和索引（首次运行），然后启动 `http://127.0.0.1:8770`。
- 端口可以用 `--port` 修改。
- README 内容：快速启动、两种模式的切换方法、五项工作与代码位置对照表（节点名 → 文件:函数）、6 条演示脚本。
