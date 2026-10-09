# Multi-Agent Research Assistant

基于 LangGraph 的专题研究系统。系统按章节执行规划、只读检索、证据分析、写作、审校、Claim 绑定和全篇审校，并通过 Checkpoint 保存每个节点的进度。

## 当前架构

当前代码包含 5 个业务角色模块：

| Agent | 职责 |
|---|---|
| `PlannerAgent` | 规划章节及章节依赖 |
| `EvidenceResearchAgent` | 使用只读检索工具执行有界多轮证据研究 |
| `SectionWriterAgent` | 撰写或修订单章正文 |
| `SectionReviewerAgent` | 审校单章支持关系、覆盖度与反证 |
| `ReportReviewerAgent` | 审查全篇冲突、重复、范围和覆盖问题 |

这些模块使用共享 Agent Runtime，拥有独立执行身份、受限上下文、结构化契约、
生命周期事件和局部 Checkpoint。EvidenceResearchAgent 拥有只读工具白名单和
有界研究循环；SectionWriterAgent 拥有确定性校验和最多三次完整重写循环。详见
[Agent Runtime 分阶段迁移基线](docs/agent-runtime-migration.md)。Claim 抽取、校验、局部修复和产物绑定已经收归
`ClaimBindingProcessor`，它是固定流程处理器，不属于 Agent。

Agent 角色模块统一使用 `*_agent.py`。`agents/contracts.py` 和
`agents/registry.py` 是契约与装配设施，不是 Agent。

检索服务本身不是 Agent。`retrieval/service.py` 负责 knowledge-service 和 Tavily
调用，`retrieval/normalization.py` 负责结果标准化与轮内去重。EvidenceResearchAgent
只能通过该只读工具边界选择检索和补充查询；全局预算与章节状态仍由工作流管理。

```text
PlannerAgent
    ↓
EvidenceResearchAgent ←→ Retrieval Service（只读非 Agent 工具）
    ↓
SectionWriterAgent
    ↓
SectionReviewerAgent ──需修订──→ 检索或重写
    ↓
ClaimBindingProcessor（非 Agent）
    ↓
下一章节 / ReportReviewerAgent
    ↓
确定性报告装配
```

系统不包含模型驱动的 Supervisor。LangGraph 使用确定性路由管理状态转换、预算、
重试和恢复。旧 Supervisor/Search/Analyst/Writer 链路已删除；旧工作流 Checkpoint
会被明确拒绝，不能自动迁移到当前图。

## Agent 与非 Agent 边界

- `agents/`：模型角色、角色 Prompt 和结构化输入输出契约。
- `processors/`：固定、可验证的数据加工流程，目前包含 Claim 绑定处理器。
- `retrieval/`：外部只读检索、响应标准化和轮内去重。
- `sections/`：章节产物、确定性校验、Claim 修补、引用和报告装配。
- `core/`：LangGraph、Checkpoint、预算、执行栅栏和流式事件。
- `runs/`：Run 状态机、持久化仓储和后台执行。
- `knowledge/`：knowledge-service 的窄 HTTP 客户端。

`workflow.py` 只负责编排：更新章节状态、保存局部成果并选择下一节点。
Prompt 和模型调用属于对应 Agent 或 Processor；确定性业务规则属于 `sections`
和 `processors`。

`agents/spec.py`、`agents/context.py`、`agents/events.py` 和 `agents/runtime.py`
已经提供独立 Agent Runtime 内核；五个业务 Agent 均已接入。Agent 执行记录、生命周期
事件及受限 JSON Checkpoint 已保存到 Run Repository，并受 execution_id 栅栏保护；
LangGraph 已升级为 v5 父图加单章节子图。五个业务 Agent 已具备固定行为评测样本，模型超时、检索
超时、暂停、预算、依赖失败、Checkpoint 后崩溃恢复和过期执行已有故障注入基线。
当前先完成旧链路清理和人工运行验收；只有人工验收通过后，才研究章节 DAG 并行。

Agent 工具必须通过 Runtime 白名单以异步方式调用。只读 Trace 接口
`GET /api/runs/{run_id}/agent-trace` 仅返回安全元数据；不会返回 Prompt、工具参数、
检索结果、候选草稿或 Agent Checkpoint 内容。

## knowledge-service 边界

本项目不直接访问或修改知识库目录，只通过 knowledge-service 执行健康检查和检索。
客户端不提供 ingestion、缓存失效、管理或其他写接口。所有检索请求固定发送
`use_query_cache=false`，调用方不能开启服务侧查询缓存写入。

联网搜索使用 Tavily；未配置 `TAVILY_API_KEY` 时自动跳过 Web 来源。

## Checkpoint 与恢复

- 每个图节点完成后由 LangGraph Checkpoint 保存状态。
- EvidenceResearchAgent 在节点内部执行有界多轮检索，并通过 Run Repository 保存局部 Checkpoint。
- 中断恢复会创建新的 Agent 执行并链接上一条 `agent_run_id`，已完成的同输入结果不会重复调用模型或检索。
- 章节、草稿、已通过的 Claim 和检索回执可独立恢复。
- 单条成功检索会持久化；同批其他查询失败时，恢复会复用成功结果。
- Claim 修复保存已通过项，只重试 pending 项。
- 当前状态版本为 `workflow_version=5`；v4 及更早的工作流 Checkpoint 不再支持。
- 每次 `section_cycle` 只处理一个章节，完成后回到父图固化产物，再进入下一章。
- 子图暂停或失败时，父图保存 `section_cycle`，子图命名空间保存具体待恢复节点。

详细说明：

- [Agent Runtime 分阶段迁移基线](docs/agent-runtime-migration.md)
- [Agent 行为评测与故障注入基线](docs/agent-evaluation.md)
- [章节子图与恢复边界](docs/section-subgraph.md)
- [人工冒烟测试清单](docs/manual-smoke-test.md)
- [运行与恢复可靠性](docs/runtime-reliability.md)
- [逐查询恢复](docs/retrieval-recovery.md)
- [模型输出恢复](docs/model-output-recovery.md)
- [章节局部操作](docs/section-operations.md)
- [研究预算](docs/run-budget.md)

## 目录结构

```text
multi_agent_research/
├── agents/
│   ├── planner_agent.py
│   ├── evidence_research_agent.py
│   ├── section_writer_agent.py
│   ├── section_reviewer_agent.py
│   ├── report_reviewer_agent.py
│   ├── contracts.py
│   ├── spec.py
│   ├── context.py
│   ├── events.py
│   ├── runtime.py
│   └── registry.py
├── processors/
│   └── claim_binding.py
├── retrieval/
│   ├── models.py
│   ├── normalization.py
│   └── service.py
├── sections/
│   ├── workflow.py
│   ├── subgraph.py
│   ├── models.py
│   ├── validation.py
│   ├── claim_repair.py
│   ├── artifacts.py
│   └── rendering.py
├── core/
│   ├── graph.py
│   ├── state.py
│   ├── streaming.py
│   ├── retrieval.py
│   ├── budget.py
│   └── execution_fence.py
├── knowledge/
│   └── client.py
├── eval/
│   ├── behavior.py
│   └── faults.py
├── runs/
└── api/
```

## 安装

项目统一使用 `multi-agent` Conda 环境：

```powershell
conda run -n multi-agent python -m pip install -e . --group dev
```

运行时、开发和评测依赖由 `pyproject.toml` 管理。本项目不依赖本地 Chroma、
PyTorch 或本地嵌入模型。

## 配置

复制 `.env.example` 为 `.env`，不要提交密钥：

```ini
DEEPSEEK_API_KEY=your_deepseek_api_key
# OPENAI_API_KEY=your_openai_api_key

TAVILY_API_KEY=your_tavily_api_key

KNOWLEDGE_SERVICE_BASE_URL=http://127.0.0.1:8001
KNOWLEDGE_SERVICE_API_KEY=your_knowledge_service_api_key
KNOWLEDGE_SERVICE_TIMEOUT=60
KNOWLEDGE_SERVICE_TOP_K=3
KNOWLEDGE_SERVICE_RETRIEVAL_MODE=hybrid

POSTGRES_DB_URL=postgresql://...
CHECKPOINT_BACKEND=sqlite
CHECKPOINT_DB_PATH=./checkpoints/research.sqlite
```

## 运行

命令行：

```python
import asyncio
from multi_agent_research.core.graph import run_research

report = asyncio.run(run_research("分析目标行业的竞争格局与主要风险"))
print(report)
```

API 和浏览器 Demo：

```powershell
conda run -n multi-agent python -m uvicorn multi_agent_research.api.server:app --host 127.0.0.1 --port 8000 --loop multi_agent_research.api.event_loop:selector_loop_factory
```

浏览器打开 `http://localhost:8000`。生产运行要求单实例、单 worker，并使用持久化
Checkpoint；断开 SSE 不会停止后台 Run。

## 测试

```powershell
conda run -n multi-agent python -m pytest -q
conda run -n multi-agent python -m ruff check multi_agent_research tests
node --test tests/test_demo_sections.cjs
```

PostgreSQL 集成测试默认跳过，显式设置 `RUN_POSTGRES_TESTS=1` 后运行。测试和诊断不得
直接打开隔壁知识库目录，也不得执行任何知识库写操作。
