# Multi-Agent Research Assistant

基于 LangGraph 的专题研究系统。系统按章节执行规划、只读检索、证据分析、写作、审校、Claim 绑定和全篇审校，并通过 Checkpoint 保存每个节点的进度。

## 当前架构

项目包含 6 个业务 Agent：

| Agent | 职责 |
|---|---|
| `PlannerAgent` | 规划章节及章节依赖 |
| `EvidenceAnalystAgent` | 判断证据充分性并提出补充检索词 |
| `SectionWriterAgent` | 撰写或修订单章正文 |
| `SectionReviewerAgent` | 审校单章支持关系、覆盖度与反证 |
| `ClaimExtractorAgent` | 建立结论、正文和来源之间的可定位关联 |
| `ReportReviewerAgent` | 审查全篇冲突、重复、范围和覆盖问题 |

Agent 角色模块统一使用 `*_agent.py`。`agents/contracts.py` 和
`agents/registry.py` 是契约与装配设施，不是 Agent。

检索不是 Agent。`retrieval/service.py` 负责 knowledge-service 和 Tavily
调用，`retrieval/normalization.py` 负责结果标准化与轮内去重。检索时机、轮次、
预算和 Checkpoint 由工作流控制；补充检索词由 `EvidenceAnalystAgent` 提供。

```text
PlannerAgent
    ↓
Retrieval Service（非 Agent）
    ↓
EvidenceAnalystAgent ──证据不足──→ Retrieval Service
    ↓
SectionWriterAgent
    ↓
SectionReviewerAgent ──需修订──→ 检索或重写
    ↓
ClaimExtractorAgent
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
- `retrieval/`：外部只读检索、响应标准化和轮内去重。
- `sections/`：章节产物、确定性校验、Claim 修补、引用和报告装配。
- `core/`：LangGraph、Checkpoint、预算、执行栅栏和流式事件。
- `runs/`：Run 状态机、持久化仓储和后台执行。
- `knowledge/`：knowledge-service 的窄 HTTP 客户端。

`workflow.py` 只负责编排：更新章节状态、控制次数、保存局部成果并选择下一节点。
Prompt 和模型调用属于对应 Agent；确定性业务规则属于 `sections`。

## knowledge-service 边界

本项目不直接访问或修改知识库目录，只通过 knowledge-service 执行健康检查和检索。
客户端不提供 ingestion、缓存失效、管理或其他写接口。所有检索请求固定发送
`use_query_cache=false`，调用方不能开启服务侧查询缓存写入。

联网搜索使用 Tavily；未配置 `TAVILY_API_KEY` 时自动跳过 Web 来源。

## Checkpoint 与恢复

- 每个图节点完成后由 LangGraph Checkpoint 保存状态。
- 章节、草稿、已通过的 Claim 和检索回执可独立恢复。
- 单条成功检索会持久化；同批其他查询失败时，恢复会复用成功结果。
- Claim 修复保存已通过项，只重试 pending 项。
- 当前状态版本为 `workflow_version=4`；更早的工作流 Checkpoint 不再支持。

详细说明：

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
│   ├── evidence_analyst_agent.py
│   ├── section_writer_agent.py
│   ├── section_reviewer_agent.py
│   ├── claim_extractor_agent.py
│   ├── report_reviewer_agent.py
│   ├── contracts.py
│   └── registry.py
├── retrieval/
│   ├── models.py
│   ├── normalization.py
│   └── service.py
├── sections/
│   ├── workflow.py
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
├── runs/
├── api/
└── tools/
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
