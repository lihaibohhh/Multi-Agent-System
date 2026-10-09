# 章节子图与恢复边界

当前 `workflow_version=5` 将章节执行从父图的平铺节点收归到
`sections/subgraph.py`。父图只管理规划、章节单元、全篇审校和报告装配：

```text
plan_sections
      │
      ▼
section_cycle（一次只处理当前章节）
      │
      ├─ 下一章 ────────────────┐
      │                         │
      ├─ report_review          │
      └─ assemble_report        │
                                │
      ◀─────────────────────────┘
```

章节子图内部执行：

```text
section_dispatch
  -> section_search（EvidenceResearchAgent）
  -> section_write（SectionWriterAgent）
  -> section_review（SectionReviewerAgent）
  -> section_claims / section_claim_gate（ClaimBindingProcessor）
  -> section_advance
  -> 返回父图
```

章节审校可以在子图内路由回 `section_search` 或 `section_write`。选章继续且已有证据时，
EvidenceResearchAgent 首轮只分析持久化证据；仍有缺口且预算允许时，才在下一轮调用只读
检索工具。原 `section_analyze` 顶层兼容节点已删除。

## 状态所有权

`SectionSubgraphState` 仅包含：研究问题、父任务只读上下文、版本、章节产物、当前章节、
章节策略、局部路由游标以及累计用量。以下状态仍由父图管理：

- `report_review` 与 `report_quality`；
- `writer_status` 与 `final_report`；
- 全局 Run 生命周期、持久化预算和 execution fence。

子图每次只处理一个当前章节。`section_advance` 完成后立即返回父图，父图提交该章产物，
再决定启动下一次 `section_cycle`。这样即使后续章节失败，已完成章节仍位于父图 Checkpoint。

## Checkpoint 与恢复

- 父图待执行节点为 `section_cycle`；
- 子图 Checkpoint 记录具体待恢复节点，例如 `section_write` 或 `section_claim_gate`；
- 恢复读取时合并父图状态和最深的待执行子图状态，避免用父图的旧投影覆盖最新章节进度；
- 流式执行启用 `subgraphs=True`，继续发布内部章节进度事件；
- 暂停或崩溃后使用 `None` 输入恢复原子图任务，不重新规划章节；
- 子图结束后父图才进入下一章、全篇审校或局部操作装配。

图拓扑和 Checkpoint 命名空间已经变化，因此版本从 v4 升级到 v5。v4 在途任务会被明确
拒绝恢复，不会被当作 v5 静默重跑。
