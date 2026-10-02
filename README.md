# Multi-Agent Research Assistant

基于 **LangGraph** 构建的专题研究系统。新任务采用章节计划、逐章检索、分析、写作与审校，最后统一装配报告。支持 **FastAPI + SSE 实时进度流**、章节草稿持久化和故障恢复，可通过浏览器 Demo 页面或 API 使用。旧版 Supervisor 节点保留，用于兼容历史 Checkpoint。

---

## 架构概览

新任务默认 `workflow_version=2`，不再让 Writer 一次生成整篇报告：

```text
章节计划 → 本章检索 ⇄ 证据分析 → 本章写作 ⇄ 本章审校
                ↑                            │
                └──────── 有限补证 ──────────┘
                              ↓
                     下一章（串行）→ 全篇装配
```

每章保存问题、证据摘录、草稿版本、引用映射、审校结论与未解决问题。后续章节只接收前章的简短交接；综合章还会接收前章选用的证据摘录，不把摘要当成证据。各阶段有独立 Checkpoint；例如第二章写作失败后，可以从第二章继续，而不重写第一章。最终装配由代码完成，不再让模型重新改写整篇。

阶段一默认最多 4 章，每章最多 2 轮证据收集、初稿后最多 1 次修订；仍串行运行，不是并发调度器。引用检查校验编号存在，不代表证明每项结论为真。详见 [阶段说明与后续计划](docs/chapter-research-roadmap.md)。

### 旧版流程（历史任务兼容）

没有版本字段的旧任务仍使用 Supervisor 路由。以下旧节点说明用于理解历史任务，不是新任务的执行路径。

```
[START]
   │
   ▼
[supervisor] ──→ [search_agent]   ──┐
             ──→ [analyst_agent]  ──┤──→ [supervisor] ──→ ... ──→ [END]
             ──→ [writer_agent]   ──┘
             ──→ [END]
```

### 节点职责

| 节点 | 模型 | 职责 |
|---|---|---|
| `supervisor` | `deepseek/deepseek-chat` | 任务分解、路由决策、终止判断 |
| `search_agent` | 无 LLM | knowledge-service 检索 + 联网搜索，写入 `search_results` |
| `analyst_agent` | `deepseek/deepseek-chat` | 批判性审查检索结果，写入 `analyst_verdict` |
| `writer_agent` | `deepseek/deepseek-chat` | 生成报告正文，并由代码确定性追加参考来源 |

---

## 状态设计（`ResearchState`）

章节流程新增 `sections`、`active_section`、`section_policy`、`section_step` 和 `report_quality`；`SectionRecord` 是单章的持久化产物，`previous_drafts` 同时保留旧正文和旧引用映射。阶段函数更新当前章节，不往 `messages` 追加整段研究历史。下图展示保留的旧流程字段。

所有节点共享一个 `ResearchState`；旧 Agent 通过约定分工写入各自字段。

```
ResearchState
├── 全局只读
│   └── research_question       用户原始问题
├── Supervisor 控制字段
│   ├── task_plan               子任务列表
│   ├── next_agent              下一跳节点名
│   ├── task_status             子任务状态字典 (pending / pass / revise / reject)
│   ├── iteration_count         当前迭代轮次
│   └── token_budget_used       累计 Token 消耗
├── 消息总线
│   └── messages                LangGraph add_messages 自动追加
├── Agent 私有工作区
│   ├── search_results          Search Agent 写入（SearchResult 列表）
│   ├── analyst_verdict         Analyst Agent 写入（AnalystVerdict 结构体）
│   └── writer_status           Writer Agent 写入（not_started / complete）
├── 事件总线
│   └── events                  各 Agent 写入结构化事件（追加 Reducer）
└── 最终输出
    └── final_report            LLM 正文 + 代码生成的参考来源节
```

### Run 与 Checkpoint 隔离

对外使用 `run_id` 表示一次独立研究任务；内部将它一对一映射为 LangGraph
`thread_id`：

```python
config = {"configurable": {"thread_id": run_id}}
```

新任务必须使用新的 `run_id`。创建接口会拒绝复用已有的
`run_id`，避免 `messages`、`search_results`、`events` 等累积字段跨任务污染。

持久化分为两层：

- `research_runs` / `research_run_events` 保存业务状态、成品、章节最新快照和可回放事件。
- LangGraph Checkpoint 保存执行位置和内部 state，用于同一 `run_id` 的故障恢复；恢复时会用其章节快照同步业务展示。两者不是一个跨存储事务。

任务从 `created → running → completed | failed | interrupted`显式流转。
`resume` 会向 LangGraph 传入 `None` 从已有 Checkpoint 继续，不会重新提交
`initial_state()`。

### Session 与父子 Run

`session_id` 是业务分组，一个 Session 可以包含多个拥有独立
Checkpoint 的 Run。未指定 Session 时，创建 Run 会自动创建一个 Session。

子 Run 可指定已完成的 `parent_run_id`，但父子 Run 必须在同一 Session。
系统会在子 Run 创建时固化 `ParentContextSnapshot`：父问题、报告摘要片段、
参考来源片段、截断标记和捕获时间。子 Run 只读取该快照，不复制父 Run
的 `messages`、`search_results`、`events` 或 Checkpoint。

### 关键子结构体

**`SearchResult`**

| 字段 | 类型 | 说明 |
|---|---|---|
| `query` | `str` | 实际执行的检索 query |
| `source` | `"knowledge" \| "web"` | 来源类型 |
| `content` | `str` | 检索到的文本内容 |
| `score` | `float` | 相关性分数（0–1） |
| `metadata` | `dict` | 原始文档元信息（`source` 文件路径、`page` 页码、`chunk_id`、`industry`、`url` 等） |
| `iteration` | `int` | 所属迭代轮次 |

> `metadata["source"]` 是完整文件路径（如 `"教育\\2025AI赋能教育行业.pdf"`），注意与 `SearchResult.source`（来源类型字符串）区分。

**`AnalystVerdict`**（Supervisor 路由的核心依据，**不解析自然语言 messages**）

| 字段 | 类型 | 说明 |
|---|---|---|
| `verdict` | `"pass" \| "revise" \| "reject"` | 审查结论 |
| `reason` | `str` | 人读解释 |
| `specific_gaps` | `list[str]` | 具体缺口，Supervisor 据此指导下轮检索 |
| `confidence_score` | `float` | 置信分（0–1，pass 建议阈值 0.75） |

**`SupervisorDecision`**（`with_structured_output` 强制格式）

| 字段 | 类型 | 说明 |
|---|---|---|
| `next` | `"search_agent" \| "analyst_agent" \| "writer_agent" \| "FINISH"` | 下一个节点 |
| `instruction` | `str` | 给下游 Agent 的具体指令（写入 messages） |
| `reason` | `str` | 决策原因（仅用于 debug / 日志） |

---

## 检索适配器与旧版 Agent 实现细节

新章节流程复用下述 Search Agent；旧 Analyst/Writer 仍用于历史 Checkpoint。新版分析、写作和审校逻辑位于 `sections/workflow.py`，每章最多选择 15 条证据，并对模型输入中的摘录长度作限制。

### Search Agent

**双路并行检索**：每轮生成 `[research_question] + analyst_verdict.specific_gaps[:2]` 共最多 3 条 query，每条同时触发 knowledge-service + Web 双路（`asyncio.gather` 并行），结果按 `score` 降序排列，按 `content[:200]` 去重后追加到 `search_results`。

**知识库边界**：本项目不打开 Chroma、不加载 Embedding/Reranker，也不接触模型缓存。所有内部知识检索都通过隔壁项目的 `knowledge-service` 完成。本项目的 HTTP 客户端只实现 `/api/v1/health/ready` 和 `/api/v1/retrieval/search`，不实现 ingestion、缓存失效或管理写接口；建库、模型与缓存生命周期由隔壁项目独占负责。

**去重与增量计数**：每轮写入 `SearchCompleted` 事件（payload 含 `new_count`、`total_count`、`queries`），供 Supervisor `_build_context` 和 `utils/dedup` 信息增量检测使用。

**Web 检索降级**：`TAVILY_API_KEY` 未配置或库未安装时，Web 路静默跳过，knowledge-service 路照常运行。

### Analyst Agent

使用 `deepseek/deepseek-chat` + `with_structured_output(AnalystVerdict, method="json_mode")` 输出结构化审查结论。

**上下文窗口控制**：取 `search_results` 最近 10 条，每条内容截断至 500 字符。

**无检索结果时**：跳过 LLM 调用，直接返回 `verdict="revise"`，`specific_gaps=[research_question]`，不消耗 token。

### Writer Agent

使用 `deepseek/deepseek-chat`，取 `search_results` 按 score 排序后的 top 15 条（每条截断至 600 字符）生成报告。

**System Prompt 关键约束**：

- 第一行必须是报告标题，**禁止**任何开场白或前言（如"好的，作为…"）
- 每条具体数据或判断必须在行内标注 `[来源N]`，不得省略
- **禁止**捏造检索结果中不存在的内容

**两段式输出**：

| 字段 | 内容 |
|---|---|
| `report_body` | Writer 节点内的临时正文（含行内 `[来源N]`，不写入 State） |
| `final_report` | `report_body` + 代码生成的 `## 参考来源` 节 |

参考来源列表由 `_build_reference_section()` 在代码侧确定性生成，不依赖 LLM，保证编号与正文 `[来源N]` 完全对应。每条来源从 `metadata` 中提取文档标题（优先 `metadata["source"]` 文件路径 → `chunk_id` 前段 → `url`）、页码（`metadata["page"]`）和行业标签（`metadata["industry"]`）。

**报告结构**：

```
## 研究报告：[标题]
### 执行摘要
### 背景
### 主要发现
   #### 发现 1：...（含 [来源N] 行内标注）
   #### 发现 2：...
### 综合分析
### 局限性与不确定性
### 结论
---
## 参考来源
[来源1] 本地知识库 | 文档标题 | p.14 | 教育 | 相关性 0.96
[来源2] 联网检索 | https://... | 相关性 0.88
```

**降级处理**：无检索结果时不调用 LLM，写入结构化错误报告；LLM 调用失败时 `try/except` 捕获，返回错误报告而非抛出异常。

---

## FastAPI + SSE 实时接口

`multi_agent_research/api/server.py` 提供 HTTP 接口，支持通过浏览器或任意 HTTP 客户端实时接收多 Agent 研究进度。

### 启动

先确认隔壁服务已健康（当前映射为 `127.0.0.1:8001`）：

```powershell
Invoke-RestMethod http://127.0.0.1:8001/api/v1/health/ready
```

```powershell
conda run -n multi-agent python -m uvicorn multi_agent_research.api.server:app `
  --host 0.0.0.0 --port 8000 `
  --loop multi_agent_research.api.event_loop:selector_loop_factory
```

浏览器打开 `http://localhost:8000` 即可看到 Demo 页面，无需启动额外的前端服务。
Demo 已从 `server.py` 中拆出：FastAPI 只负责挂载 `api/static/` 资源，页面结构、
样式和交互分别位于 `index.html`、`styles.css` 和 `app.js`。

页面支持重新加载历史 Session、父子 Run 树、历史报告查看、基于已完成 Run 创建
子任务，以及对 `failed/interrupted` Run 使用原 `run_id` 从 Checkpoint 恢复。

### 接口

| 路径 | 说明 |
|---|---|
| `GET /` | 独立静态 Demo（Session/Run 树与实时 Agent Pipeline） |
| `POST /api/sessions` | 显式创建 Session |
| `GET /api/sessions/{session_id}` | 读取 Session 和按时间排列的 Run 时间线 |
| `GET /api/sessions/{session_id}/runs` | 列出 Session 内的 Run |
| `POST /api/runs` | 创建 Run，此时不执行 |
| `POST /api/runs/{run_id}/start` | 后台启动 `created` Run |
| `POST /api/runs/{run_id}/resume` | 从 Checkpoint 恢复 `failed/interrupted` Run |
| `GET /api/runs/{run_id}` | 读取状态、错误或最终报告，不重复执行 |
| `GET /api/runs/{run_id}/stream` | 回放并追踪持久化 SSE 事件，支持 `after` 游标 |
| `GET /api/research/stream` | 旧的一步式 SSE 兼容接口（已弃用） |
| `GET /api/health` | 健康检查 |
| `GET /api/docs` | Swagger 文档 |

### SSE 事件序列

客户端创建并启动 Run 后，通过 `EventSource` 连接
`/api/runs/{run_id}/stream`。事件先写入 PostgreSQL 再推送，因此断开 SSE
不会中止任务，重连可从指定 sequence 继续读取。

| event | 关键字段 | 说明 |
|---|---|---|
| `start` | `question`, `run_id` | 新任务开始确认 |
| `section_plan` | `sections` | 新流程的章节计划 |
| `section_progress` | `stage`, `sections` | 章节阶段产物与完整章节快照 |
| `section_snapshot` | `sections` | 恢复时同步 Checkpoint 中的章节产物 |
| `supervisor_decision` | `run_id`, `iteration`, `next`, `reason` | 每轮路由决策 |
| `search_complete` | `new_count`, `total_count`, `queries` | 检索轮次完成 |
| `analyst_verdict` | `verdict`, `confidence`, `gaps` | 审查结论 |
| `report_ready` | `char_count`, `preview` | 报告生成完成 |
| `done` | `report`, `char_count`, `total_iterations`, `total_results` | 全部结束，含完整报告 |
| `error` | `message`, `type` | 任务异常 |

### 流式架构

```
multi_agent_research/core/streaming.py
├── _get_app()              Graph 异步单例（使用配置的 Checkpointer）
├── _parse_supervisor()     解析 supervisor 节点输出
├── _parse_search()         解析 search_agent 节点输出（读 SearchCompleted 事件）
├── _parse_analyst()        解析 analyst_agent 节点输出（兼容 dict / Pydantic model）
├── _parse_writer()         解析 writer_agent 节点输出
├── astream_research()      执行全新 Run
└── aresume_research()      从已有 Checkpoint 继续 Run
```

`_get_app()` 通过异步锁与模块级缓存保证 Graph 只编译一次。FastAPI
启动时会初始化 Session / Run / Event 表、将上次进程遗留的 `running` 转为 `interrupted`，
并检查 knowledge-service 就绪状态。

---

## LLM 配置

### `load_chat_model(model_ref)` 接口

`multi_agent_research/utils/llm.py` 对外暴露唯一入口，接受 `"provider/model-name"` 格式字符串，结果按 `model_ref` 缓存（`@lru_cache(maxsize=32)`）。

**支持的 provider**：

| provider 别名 | 实际后端 | 配置来源 |
|---|---|---|
| `openai` | `ChatOpenAI` | `settings.openai` |
| `anthropic` | `ChatAnthropic` | `settings.anthropic` |
| `deepseek` / `ds` | `ChatOpenAI`（兼容接口） | `settings.deepseek` |
| `local` / `openai-compatible` | `ChatOpenAI`（兼容接口） | `settings.local.base_url` / `LOCAL_LLM_BASE_URL` |

DeepSeek 默认 `base_url` 为 `https://api.deepseek.com`，可通过 `DEEPSEEK_BASE_URL` 改为 OneAPI 或代理地址。

---

## 终止条件

新章节流程使用计划中固化的 `SectionPolicy` 限制每章证据收集轮次和修订次数，配置见 `.env.example` 的 `AGENT_SECTION_*`。证据不足的章节以 `limited` 状态和显式局限性完成；超过修订次数后仍有非法引用编号则任务失败，不发布成品。模型调用异常或截断也会失败并保留此前 Checkpoint。

`token_budget_used` 在新流程中记录成功模型调用已返回的 Token 用量，`usage_unknown_calls` 记录缺少用量信息的成功调用；失败或底层自动重试的费用可能不在其中，因此不是严格账单或硬性全局 Token 限额。下面的 `MAX_ITERATIONS` / `TOKEN_BUDGET` 仅适用于旧版 Supervisor。

### 硬性终止（`should_terminate()`）

在 Supervisor LLM 调用**之前**执行，任意一条触发即终止：

| 条件 | 触发值 | 常量 |
|---|---|---|
| 迭代超限 | `iteration_count >= 6` | `MAX_ITERATIONS` |
| Token 预算耗尽 | `token_budget_used >= 500000` | `TOKEN_BUDGET` |

### 信息增量检测（`utils/dedup.detect_information_gain()`）

防止"搜索枯竭但仍反复触发 search_agent"的死循环，两级算法顺序执行：

| 级别 | 算法 | 触发条件 |
|---|---|---|
| **Level 0** | 事件计数 | 连续 2 轮 `new_count == 0` |
| **Level 1** | 字符 bigram Jaccard | 相邻两轮内容相似度 > 0.85 |

数据不足或结构异常时一律返回"有新信息"（保守策略）。

### 双层路由保护

```
Layer 1（supervisor_node 内部）：LLM 调用前检查，省 token
Layer 2（route_from_supervisor）：LLM 返回后再次校验，兜底异常状态
```

---

## 信息增量检测（`utils/dedup.py`）

```python
from multi_agent_research.utils.dedup import detect_information_gain, GainCheckResult

result: GainCheckResult = detect_information_gain(
    events=state.get("events", []),
    search_results=state.get("search_results", []),
).log()  # 链式调用，打印日志并返回自身

if not result:
    # 信息枯竭，强制推进 writer 或终止
    ...
```

**中文安全**：使用字符级 bigram 替代空格分词，`"深度学习"` → `{"深度", "度学", "学习"}`，无需 jieba。

**切片逻辑**：利用事件 payload 中的 `total_count` / `new_count` 定位每轮新增结果在 `search_results` 列表中的位置，不依赖 `SearchResult.iteration` 字段。

---

## 项目结构

```
.
├── multi_agent_research/
│   ├── agents/
│   │   ├── analyst_agent.py      # 审查 Agent
│   │   ├── search_agent.py       # 检索 Agent（knowledge-service + Tavily 并行）
│   │   └── writer_agent.py       # 写作 Agent（两段式输出 + 代码侧参考来源生成）
│   ├── api/
│   │   ├── __init__.py
│   │   ├── demo.py               # Demo 首页与静态资源挂载
│   │   ├── server.py             # FastAPI 生命周期、业务 API 与 SSE 接口
│   │   └── static/
│   │       ├── index.html        # Demo 页面结构
│   │       ├── styles.css        # Demo 响应式样式
│   │       └── app.js            # Session/Run 树、恢复、SSE 与报告交互
│   ├── core/
│   │   ├── checkpointer.py         # LangGraph Checkpoint 后端工厂
│   │   ├── graph.py              # 图组装、编译与运行入口
│   │   ├── run_context.py        # run_id 校验与 thread_id 一对一映射
│   │   ├── state.py              # ResearchState、追加/去重 Reducer、initial_state
│   │   ├── config.py             # pydantic-settings 全局配置单例
│   │   ├── streaming.py          # 新建/恢复图执行与语义事件转换
│   │   └── supervisor.py         # Supervisor 节点逻辑与路由决策
│   ├── knowledge/
│   │   └── client.py             # knowledge-service 只读 HTTP 客户端
│   ├── runs/
│   │   ├── models.py             # Session / Run 状态机与父上下文契约
│   │   ├── repository.py         # PostgreSQL Session / Run / Event 仓储
│   │   └── service.py            # 分组、继承、后台执行、恢复与 SSE 解耦
│   ├── sections/
│   │   ├── models.py             # 章节计划、产物、版本与次数限制
│   │   ├── workflow.py           # 逐章检索、分析、写作、审校和推进
│   │   └── rendering.py          # 来源去重、引用验证与整篇装配
│   ├── tools/
│   │   ├── knowledge.py          # knowledge-service 工具适配器
│   │   ├── search.py             # Tavily Web 搜索工具适配器
│   │   └── support.py            # 结果信封、裁剪和重试策略
│   └── utils/
│       ├── dedup.py              # 信息增量检测（GainCheckResult / detect_information_gain）
│       ├── llm.py                # LLM 加载与 provider 解析
│       └── redis_client.py       # Redis 连接池单例
├── scripts/
│   ├── smoke_test_llm.py         # 不在代码中嵌入密钥的 LLM 冒烟测试
│   ├── e2e_run_lifecycle.py      # 真实 Run 生命周期端到端验收
│   ├── compare_analyst_models.py # DeepSeek / 本地 vLLM 对比实验
├── tests/
│   ├── test_graph.py             # 图路由和状态测试
│   ├── test_knowledge_client.py  # knowledge-service HTTP 契约测试
│   ├── test_run_service.py       # Run 状态流转与 SSE 解耦测试
│   ├── test_run_repository_postgres.py # PostgreSQL 往返集成测试
│   └── test_tool_support.py      # 工具协议与重试测试
├── checkpoints/
├── .env
├── pyproject.toml               # 项目元数据、运行/开发/评测依赖与工具配置
└── README.md
```

---

## 测试

### 单元测试

```powershell
conda run -n multi-agent python -m pytest -q

# 仅验证工具结果协议、裁剪和重试，不访问 LLM / Redis / 外网
conda run -n multi-agent python -m unittest tests.test_tool_support -v
```

| 测试文件 | 覆盖对象 |
|---|---|
| `test_graph.py` | 终止条件、Supervisor 路由和初始状态 |
| `test_sections.py` | 章节隔离、有限修订、引用装配、SQLite 重启恢复和旧版兼容 |
| `test_knowledge_client.py` | knowledge-service 鉴权、健康检查、检索响应与错误分类 |
| `test_run_service.py` | 创建/执行分离、完成、中断、恢复与 SSE 断开语义 |
| `test_run_repository_postgres.py` | PostgreSQL Run 与事件持久化（默认跳过） |
| `test_tool_support.py` | 工具结果裁剪、Tavily 响应归一化和重试元数据 |

```powershell
# 显式运行 PostgreSQL 集成测试（会创建并清理唯一的测试 Run）
$env:RUN_POSTGRES_TESTS='1'
conda run -n multi-agent pytest -q tests/test_run_repository_postgres.py
```

### 冒烟测试

```bash
python -m multi_agent_research.core.graph

# 只调用一次已配置的模型
python -m scripts.smoke_test_llm --model-ref deepseek/deepseek-chat

# DeepSeek 与本地 vLLM 对比（需先启动 vLLM）
python -m scripts.compare_analyst_models
```

`scripts/sample_results_rich.json` 和 `sample_results_poor.json` 是人工构造的评测样例，不是事实依据。真实检索样例 `sample_search_results.json` 只保留在本地，不随公开仓库分发；对比脚本在缺少该文件时跳过对应场景。

---

## 快速开始

### 1. 安装依赖

```powershell
# 项目统一使用 multi-agent Conda 环境
conda run -n multi-agent python -m pip install -e . --group dev

# 如需运行离线评测，再安装 eval 依赖组
conda run -n multi-agent python -m pip install -e . --group dev --group eval
```

所有运行时、开发和评测依赖均由 `pyproject.toml` 统一管理，不再维护
`requirements.txt`。本项目不再依赖 PyTorch、Hugging Face、Chroma 或本地检索模型，
因此不存在 GPU/CPU 推理配置；检索算力由 `knowledge-service` 统一承担。

### 2. 配置环境变量

从 `.env.example` 复制为本地 `.env` 后填写密钥；不要提交 `.env`。API 的 Run 存储需要本项目自己的 PostgreSQL（`POSTGRES_DB_URL`）；Checkpoint 可单独选 SQLite 或 PostgreSQL。启动时会为 `research_runs` 增加 `sections JSONB` 列，升级前请备份本项目数据库。不会修改隔壁知识库。

```ini
# .env

# ── LLM（至少一个必填）────────────────────────
DEEPSEEK_API_KEY=your_deepseek_api_key
OPENAI_API_KEY=your_openai_api_key

# ── 联网搜索（建议配置；未配置时 Web 路静默跳过）──
TAVILY_API_KEY=your_tavily_api_key

# ── 隔壁知识库服务（必填）──────────────────────
KNOWLEDGE_SERVICE_BASE_URL=http://127.0.0.1:8001
KNOWLEDGE_SERVICE_API_KEY=your_knowledge_service_api_key
KNOWLEDGE_SERVICE_TIMEOUT=60
KNOWLEDGE_SERVICE_TOP_K=3
KNOWLEDGE_SERVICE_RETRIEVAL_MODE=hybrid

# ── DeepSeek 代理（可选）──────────────────────
# DEEPSEEK_BASE_URL=https://api.deepseek.com

# ── LangSmith（可选；未配置不影响运行）──────────
# LANGCHAIN_TRACING_V2=true
# LANGCHAIN_API_KEY=your_langsmith_api_key
```

本项目直接在 Windows 上运行时使用 `127.0.0.1:8001`；若本项目也放入 Docker，
将 `KNOWLEDGE_SERVICE_BASE_URL` 改为 `http://host.docker.internal:8001`（或同一
Compose 网络中的服务名）。

### 3a. 命令行调用

```python
import asyncio

from multi_agent_research.core.graph import run_research

report = asyncio.run(run_research(
    question="2026年以来国内AI教育行业有哪些重大进展？结合市场格局分析当前投资价值",
    run_id="run_ai_education_001",
))
print(report)
```

### 3b. API + 浏览器 Demo

```bash
conda run -n multi-agent python -m uvicorn multi_agent_research.api.server:app --host 127.0.0.1 --port 8000 --loop multi_agent_research.api.event_loop:selector_loop_factory
# 浏览器打开 http://localhost:8000
```

```javascript
// 创建和执行分离；即使 SSE 关闭，后台 Run 仍会继续。
const created = await fetch('/api/runs', {
  method: 'POST',
  headers: {'Content-Type': 'application/json'},
  body: JSON.stringify({question: '你的研究问题'}),
}).then(response => response.json());

await fetch(`/api/runs/${created.run_id}/start`, {method: 'POST'});
const es = new EventSource(`/api/runs/${created.run_id}/stream`);
es.addEventListener('done', e => {
    const { report, total_iterations, total_results } = JSON.parse(e.data);
    console.log(report);
});

// 后续任务：同 Session 分组，显式继承已完成的父 Run 产物快照
const child = await fetch('/api/runs', {
  method: 'POST',
  headers: {'Content-Type': 'application/json'},
  body: JSON.stringify({
    question: '基于上一份报告，哪些指标最值得跟踪？',
    session_id: created.session_id,
    parent_run_id: created.run_id,
  }),
}).then(response => response.json());
```

真实端到端验收（会调用 knowledge-service、Web 搜索和当前配置的 LLM）：

```powershell
conda run -n multi-agent python -m scripts.e2e_run_lifecycle `
  "基于现有资料，分析 AI 教育行业的主要机会与风险"
```
