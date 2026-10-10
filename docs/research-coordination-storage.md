# 研究协调工作区（Schema v1）

这一层保存父 Run 与不同章节之间需要共享的**结构化研究资产**，不保存 Agent
对话、Prompt 或完整运行日志。数据库使用规范化表存储，通过只读视图
`research_coordination_units` 向协调层呈现“父 Run / 各章节”的层次化逻辑表。

## 代码位置

- `multi_agent_research/coordination/models.py`：写入契约和协调快照模型。
- `multi_agent_research/coordination/schema.py`：PostgreSQL 表、索引和只读视图。
- `multi_agent_research/coordination/repository.py`：原子写入、版本校验和快照读取。
- `multi_agent_research/runs/repository.py`：在服务启动时安装 Schema，并公开仓储接口。

## 物理表

| 表 | 作用 | 关键字段 |
|---|---|---|
| `research_workspaces` | 一个 Run 的共享研究工作区 | `run_id`, `workspace_version`, `status`, `summary` |
| `research_coordination_scopes` | 父 Run 或章节一级单元 | `scope_type`, `scope_id`, `source_run_id`, `revision`, `summary`, `summary_claim_ids`, `summary_metric_ids`, `dependency_claim_ids`, `open_questions` |
| `research_workspace_documents` | 工作区内去重后的来源文档 | `document_id`, `canonical_url`, `title`, `source_type`, `content_hash` |
| `research_evidence` | 可定位的来源摘录 | `evidence_id`, `document_id`, `excerpt`, `locator`, `status` |
| `research_claims` | 父 Run 或章节结论 | `claim_id`, `statement`, `origin_*`, `claim_type`, `status`, `visibility`, `revision` |
| `research_claim_evidence_bindings` | 结论与引用之间的支持判断 | `claim_id`, `evidence_id`, `support_status`, `reason`, `required_supplement`, `quote_refs` |
| `research_metrics` | 带统计口径的数据 | `metric_id`, `value_*`, `unit`, `period`, `geography`, `sample_scope`, `numerator_definition`, `denominator_definition` |
| `research_metric_evidence_bindings` | 数据与来源摘录的多对多关系 | `metric_id`, `evidence_id` |

## 逻辑协调表

`research_coordination_units` 每行表示一个父 Run 或章节：

```text
scope
├─ summary
├─ dependency_claim_ids
├─ open_questions
├─ evidence[]
│  └─ document
├─ claims[]
│  └─ evidence_bindings[]
│     └─ quote_refs[]
└─ metrics[]
   └─ evidence_ids[]
```

`scope_type = 'parent_run'` 的行是父 Run 快照；`scope_type = 'section'` 的行是章节
成果。一个工作区最多只有一个父 Run scope，可以有多个章节 scope。

## 支持状态

引用支持状态不是简单的布尔值：

- `supports`：摘录直接或充分支持结论；
- `contradicts`：摘录与结论冲突；
- `insufficient`：没有明确反驳，但不足以证明结论；
- `pending`：尚未完成语义审校。

除 `supports` 外，写入时必须提供 `reason`；需要补证据时可填写
`required_supplement`。

一个 Claim 对同一 `evidence_id` 只保存一条支持关系，但该关系可以通过
`quote_refs` 保留多段原文、原章节来源编号、关系类型和字符定位。投影层先合并
同源多引文，契约层拒绝重复绑定，数据库层再以主键 upsert 作为最后的幂等保护。
若同一证据同时出现支持与反向引文，聚合状态为 `pending` 并要求后续协调审查。

## 一致性边界

- 所有写入以 `expected_workspace_version` 执行乐观锁；过期章节不能覆盖新成果。
- 文档在同一个 Run 工作区内按稳定 `document_id` 复用。
- 一个章节可用 `referenced_evidence_ids` 引用父 Run或其他章节已经提交的证据，
  无需复制摘录和来源。
- Claim、Metric 和 Evidence 使用稳定 ID 绑定，不使用最终报告中的临时引用序号。
- 数据至少保留时间、地域、单位、样本和分子/分母定义，供后续冲突判断使用。

## 当前接入状态

Schema v1、仓储读写和工作流投影已经接通：

1. 创建子 Run 时，父 Run handoff 被转换为一个不可变的 `parent_run` scope；
2. 章节达到 `complete`、`limited` 或 `stale` 后，`SectionRecord` 自动投影为
   `section` scope；
3. `SectionReview.summary` 作为章节摘要文本，同时保存其 `summary_claim_ids`；
4. 父 Run 摘要从已接受 Claim 确定性生成，不读取 `report_excerpt`；
5. 后续章节会收到不含原始摘录的有限 `ResearchBrief`；
6. 前章已接受 Claim 的证据可进入后章候选证据集，由研究 Agent 重新判断相关性；
7. 使用共享证据的章节记录上游 Claim 依赖；上游 Claim 改变后，依赖 scope 自动
   标记为 `stale`；
8. 上游撤回 Evidence 时将其标记为 `stale`，保留下游历史绑定供审计，不级联抹除；
9. Run 完成时从非 stale 的 accepted Claim 生成工作区摘要。

当前 `summary_metric_ids` 和 `research_metrics` 已具备存储、读取和校验能力，但现有
Claim 提取产物还没有结构化 Metric，因此不会从正文正则猜测指标。下一阶段应增加
受约束的指标提取/口径归一化处理器。跨章节最终编辑已经由 `ChiefEditorAgent`
接入：协调工作区继续提供共享 Claim、Metric 和开放问题，主编只消费有界快照，
最终引用编号和来源去重仍由确定性渲染器完成。后续再扩展持久化的 EditorialPlan。
