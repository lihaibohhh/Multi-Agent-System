# Agent Runtime 分阶段迁移基线

本文记录当前角色化工作流的边界、目标架构和分阶段迁移约束。它是实施基线，
不是把所有模块都命名为 Agent 的理由。

## 当前基线

当前系统使用 `workflow_version=6`，由父图和串行章节子图共同执行：

```text
plan_sections
  -> section_cycle
     -> section_search
     -> section_write
     -> section_review
     -> section_claims / section_claim_gate
     -> section_advance
  -> report_review
  -> chief_edit
  -> edited_report_review
  -> assemble_report
```

`agents/` 中当前有六个业务 Agent：

- `PlannerAgent`
- `EvidenceResearchAgent`
- `SectionWriterAgent`
- `SectionReviewerAgent`
- `ReportReviewerAgent`
- `ChiefEditorAgent`

Claim 模型抽取、确定性校验、局部修复和产物绑定属于
`processors/claim_binding.py` 中的 `ClaimBindingProcessor`，不再注册为 Agent。

其中 EvidenceResearchAgent 已拥有只读工具白名单和有界多轮行动循环；其他 Agent
通过共享 Agent Runtime 执行。LangGraph 继续负责全局章节状态和确定性路由，Agent
Runtime 负责局部轮次、生命周期、Checkpoint 与恢复链接。

## 术语和所有权

### Agent

负责需要语义判断和有限自主行动的任务。目标状态下，每个 Agent 应拥有：

- 稳定身份、职责和版本；
- 独立模型与模型参数；
- 受限输入上下文和输出契约；
- 工具白名单（允许为空）；
- 有界轮次、预算和超时；
- 输入/输出 Guardrail；
- 独立 `agent_run_id`、生命周期事件和 Trace；
- 明确的完成、暂停、失败和取消语义。

### Processor

执行固定、可验证的数据加工流程。Processor 可以在某一步调用 LLM，但下一步由代码
决定，不拥有开放式行动循环。Claim 抽取、校验、局部修复和产物绑定最终归入
`ClaimBindingProcessor`。

### Tool

向 Agent 暴露单一、受限能力。工具不决定工作流，也不拥有业务状态。知识检索和 Web
检索仍由 `retrieval/` 实现，未来仅作为 `EvidenceResearchAgent` 的只读工具。

### Orchestrator

LangGraph 编排器管理全局状态、依赖、路由、预算、Checkpoint 和恢复。项目不重新引入
模型驱动的 Supervisor。

### Service

提供持久化、外部检索、配置和运行时基础能力，例如 `runs/`、`knowledge/` 和
`retrieval/`。Service 不是 Agent。

## 目标架构

```text
LangGraph 确定性编排器
|
|-- PlannerAgent
|-- EvidenceResearchAgent
|   `-- 只读 Retrieval Tools
|-- SectionWriterAgent
|-- SectionReviewerAgent
|-- ClaimBindingProcessor
|-- ReportReviewerAgent
|-- ChiefEditorAgent
`-- ReportAssembler
```

目标是六个独立 Agent、一个 Claim Processor，以及继续由代码负责的编排、校验、
报告装配和运行时基础设施。

## 迁移阶段

1. **已完成：**固化现状、术语、测试基线和知识库访问边界。
2. **已完成：**将 `ClaimExtractorAgent` 迁入 `ClaimBindingProcessor`，保持产物和恢复语义不变。
3. **已完成：**建立 `AgentSpec`、`AgentContext`、`AgentResult`、`AgentRunner` 和
   内存生命周期事件；暂不接入生产工作流或改变 v4 State。
4. **已完成：**低风险单轮 Agent 已逐个迁移。Planner、SectionReviewer 和
   ReportReviewer 均通过 `AgentSpec`、隔离的 `AgentContext` 和 `AgentRunner`
   接入现有节点，未改变节点名称、State 或 Checkpoint 版本。
5. **已完成：**在 Run Repository 中增加 Agent 执行记录、幂等生命周期事件、受限
   JSON Checkpoint 和交接协议；所有写入受当前 `execution_id` 栅栏保护，孤儿执行会
   在对账时标记为 `interrupted`，未修改 LangGraph v4 State。
6. **已完成：**将 EvidenceAnalyst 升级为 EvidenceResearchAgent；Agent 自主管理最多
   四轮“检索、充分性审查、缺口查询、补充检索”，仅获授权调用
   `retrieval/service.py` 的只读工具。局部 Checkpoint 可跨 Run 执行批次恢复。
7. **已完成：**将 SectionWriter 升级为拥有本地写作、确定性引用校验和最多三次
   完整重写机会的 Agent；候选草稿、校验问题和尝试次数进入安全 Checkpoint，
   已完成的相同输入可直接恢复而不重复调用模型。
8. **已完成：**在扩大图结构前补齐可观测性和评测基线。
   - **已完成：**Runtime 强制使用异步工具，通过白名单包装器记录
     `agent_tool_started/completed/failed`；事件不包含参数、结果或异常原文。
   - **已完成：**提供只读安全 Trace API，只展示身份、轮次、状态、工具名、耗时、
     异常类型和用量摘要，不返回 Prompt、来源正文、local_state 或 handoff 内容。
   - **已完成：**建立六个业务 Agent 的固定行为样本，统一检查身份、终态、轮次、
     工具边界、引用、证据缺口和场景化的无依据确定性表述。
   - **已完成：**建立模型超时、检索超时、暂停、预算耗尽、依赖失败、Checkpoint 后
     崩溃恢复和过期执行栅栏的故障注入矩阵；详见 `docs/agent-evaluation.md`。
9. **已完成：**将单章节执行封装为 `section_cycle` 子图，保持章节串行；验证
   Agent/Processor 交接、内部流式事件、暂停、崩溃和 SQLite 跨进程恢复。图拓扑升级为
   v5，v4 Checkpoint 明确拒绝恢复；`section_analyze` 兼容入口已删除。
10. **已完成：**增加 `ChiefEditorAgent`、稳定 Evidence Token、编辑后独立复审和
    确定性引用渲染。父图升级为 v6，v5 及更早 Checkpoint 明确拒绝恢复。
11. **已完成：**删除孤立旧工具链和失效配置，将仍有价值的测试迁移到当前
    `retrieval/` 边界。完整自动化回归结果为 230 passed、8 个需显式启用的 PostgreSQL
    集成测试 skipped；前端 DOM 契约 16 passed，Ruff 与补丁空白检查通过。
12. **人工验收闸门：**按 `docs/manual-smoke-test.md` 验证真实服务、浏览器主流程、
    暂停恢复和最终报告。人工验收通过前不改变章节调度方式。
13. **后续候选优化：**只有人工验收通过后，才在依赖、预算、评测和执行栅栏约束下
    评估章节 DAG 并行；是否实施取决于串行基线的实际数据。

每个阶段必须独立回归并汇报；后续阶段不得依赖未验证的中间状态。

## Checkpoint 兼容策略

- 早期只调整内部所有权时保持 `workflow_version=4`。
- 章节子图曾将版本从 v4 升级为 v5；主编编辑与编辑后复审节点进一步将当前版本升级为
  `workflow_version=6`。
- v5 及更早的在途任务不得迁移到 v6 图中，恢复时会明确拒绝；用户需要创建新 Run。
- 版本升级必须明确选择迁移或拒绝恢复，不得静默重跑已有研究。

## 强制访问边界

- 不直接打开或访问 `D:\python项目集合\src\chroma_db`。
- 不修改 `D:\python项目集合\src` 的代码、配置、文档或运行数据。
- 本项目只通过 knowledge-service 健康检查和检索接口读取知识。
- 不实现 ingestion、cache invalidation、admin 或其他写接口。
- knowledge-service 检索始终固定 `use_query_cache=false`，不得成为 Agent 可修改参数。

## 阶段验收门槛

每一阶段至少需要：

- 相关单元测试和恢复测试通过；
- Python 完整测试套件不低于本基线；
- 前端 DOM 测试通过；
- Ruff 和 `git diff --check` 通过；
- Graph 节点、State Schema 或 Checkpoint 版本的变化得到明确说明；
- 没有新增知识库写入能力或直接知识库目录访问。

## 当前已知待处理项

- `ClaimBindingProcessor` 已独立于 Agent Registry；每次处理尝试仍保留原有图节点
  Checkpoint 边界。
- Agent Runtime 内核已经建立，Planner、EvidenceResearch、SectionWriter、
  SectionReviewer、ReportReviewer 和 ChiefEditor 均已接入；六个业务 Agent 的运行边界
  已经定型。父图、`sections/subgraph.py` 和 v6 Checkpoint 是权威编排路径。
- Agent 执行事件和最新 `local_state`、`handoff`、`unresolved`、usage 已独立持久化到
  Run Repository。EvidenceResearchAgent 恢复时会创建新的 `agent_run_id`，链接上一条
 执行记录并加载安全 Checkpoint；LangGraph 节点 Checkpoint 仍是全局状态提交边界。
- 旧 `tools/search.py`、`tools/knowledge.py` 和 `tools/support.py` 已由
  `retrieval/service.py` 与 `retrieval/normalization.py` 取代；仍有价值的测试已经迁移，
  旧实现和旧测试均已删除。
- DAG 并行暂不实施。当前权威基线是 v6 父图加串行单章节子图，等待人工冒烟验收。
- 当前大规模 Agent 角色迁移尚未提交 Git；提交或创建基线分支需要用户单独授权。
