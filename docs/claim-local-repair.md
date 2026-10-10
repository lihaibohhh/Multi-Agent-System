# Claim 候选、绑定与局部修复（2026-10-10）

## 当前协议

Claim 处理由 `ClaimBindingProcessor` 负责，它是可验证的固定流程，不是独立 Agent。首次抽取和局部修复现在统一使用 V3 不透明片段 ID：

1. 程序对当前版本的章节正文和来源摘录切片，向模型只暴露 `D0001`、`E0001` 一类目录键、文本和来源号，不暴露字符偏移。
2. 首次抽取输出 `ClaimCandidateBatch`；模型只能提交结论语义、`draft_segment_id`、assessment、caveat 和单层证据 `segment_id`。每条 Claim 最多 8 个 evidence 项，Schema、提示词和持久化上限一致。
3. 修复输出 `ClaimRepairBatch`；模型只能修复 pending slot 的片段选择和支持判断，不能重写原 statement。
4. 两个输入 Schema 都使用 `extra="forbid"`，并且根本不包含 `claim_id`、`evidence_id`、`draft_span`、`quote_span`、原文 quote 等程序所有字段。模型即使输出这些字段，也会在输入边界被拒绝。
5. 程序校验 ID 是否属于当前目录、D/E 类型是否正确、是否重复，再从内部目录回填精确原文和 `[start, end)`，构造持久化 `Claim`/`EvidenceLink`。

这个边界保持“模型负责语义选择，程序拥有身份与原文定位”。输出结构仍是原有 `Claim`，因此章节审查、跨章共享、`ChiefEditorAgent` 和报告装配无需改用新字段。

## 切片与版本

V3 目录按句末标点和空行分段，将 PDF 单换行视为软换行，避免一段完整论述被切成大量几十字的碎片。内部仍保留原文的精确连续偏移，模型目录才将单换行显示为空格。单段最多 450 字符，每条来源使用前 1000 字符。

目录指纹包含切片版本、章节 ID/修订号、正文和可见来源文本。正文、来源或切片版本变化时，旧 ID 不能直接继承。未来改用更强的句子/段落切分器时，应提升 `SEGMENTATION_VERSION`，而不是静默改变目录含义。

## 部分接受、重试与恢复

- 每个候选独立校验；合法兄弟项立即保存，失败项记录稳定 slot 和错误，之后只修复 pending。
- 已接受项不允许被补丁覆盖；原结论不允许偷换，已知反证不能删除，uncertain/unsupported 不能借定位修复升级。
- 同一工作包最多 3 次 Claim 调用；连续两次得到同一失败签名时可提前停止。`attempts`、`total_attempts` 和停滞签名持久化到 Checkpoint。
- 普通 `/resume` 不再因为 execution ID 变化而重置额度，因此无法通过反复恢复绕过上限。到限后 Run 保留通过项并暂停，用户需要补证据、刷新来源、重写章节或人工处置。
- 旧 V1/V2 Checkpoint 在首次加载时迁移：保留已接受 Claim 和累计用量，删除 pending 中已过期的片段 ID/偏移，为 V3 新策略开放一个有界尝试窗口。迁移完成后不会因后续 Run 重新开窗。

## 安全边界与局限

精确定位通过只说明文本确实存在于当前快照，不代表证据真实、完整或真正支持 Claim。assessment 和 relation 仍是模型的语义判断，章节审查和全篇审校仍需保留不确定性。

本项目不直接读写隔壁的 `chroma_db`；Claim 协议升级不改变知识库边界，仍只能通过 `knowledge-service` 以 `use_query_cache=false` 检索。

## 验证入口

- 协议与迁移：`tests/test_claim_protocol_v2.py`
- 局部修复：`tests/test_claim_repair.py`
- Processor 边界：`tests/test_claim_binding_processor.py`
- 图恢复与显式刷新：`tests/test_section_revisions.py` 和 `tests/test_business_validation.py`

部署后建议手工验收三类案例：多片段证据、同时含支持/反证、无可用证据。历史暂停 Run 首次加载会迁移为 V3，但不会自动写入数据库；只有实际恢/操作该 Run 时才会随新 Checkpoint 持久化。
