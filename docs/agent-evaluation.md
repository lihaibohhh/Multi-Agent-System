# Agent 行为评测与故障注入基线

本基线在不调用真实模型、不联网、不访问本地知识库目录的前提下，对六个业务 Agent
执行可重复的行为回归。它验证的是 Agent 边界和已明确编码的业务约束，不宣称通过
固定样本即可证明任意模型输出都没有事实错误。

## 评测结构

```text
固定请求、固定模型响应、内存只读工具
                 │
                 ▼
             AgentRunner
                 │
        AgentResult + AgentEvent[]
                 │
                 ▼
       evaluate_behavior()
                 │
       通用运行时规则 + 场景规则
                 │
                 ▼
       EvaluationSuiteReport
```

`multi_agent_research/eval/behavior.py` 提供：

- `BehaviorObservation`：一次执行的结果、异常和生命周期事件；
- `BehaviorExpectation`：预期终态、最大轮次、工具白名单和事件要求；
- `BehaviorCheck`：场景特有的确定性判定；
- `BehaviorEvaluation` 与 `EvaluationSuiteReport`：可序列化评测结果；
- `observe_agent_run()` 和 `evaluate_behavior()`：执行观测与判定入口。

所有场景共同检查：

- Agent 身份及 `agent_run_id` 一致；
- 生命周期从唯一 `agent_started` 开始；
- 完成、暂停、失败和中断终态符合预期；
- turn 连续且不超过 `AgentSpec.max_turns`；
- 模型调用次数符合场景约束；
- 工具没有越过白名单；
- 工具开始与完成/失败事件正确配对；
- 工具 Trace 不包含参数、结果或异常原文；
- 生命周期 `event_id` 不重复。

## 六个业务 Agent 基线

| Agent | 固定场景 | 重点判定 |
|---|---|---|
| PlannerAgent | 带受限父章节的规划 | 章节计划只引用提供的父章节 |
| EvidenceResearchAgent | 首轮发现缺口并补充检索 | 缺口进入下一轮查询、工具受限、轮次有界 |
| SectionWriterAgent | 首稿引用越界后本地重写 | 最终引用有效、限制得到披露、没有样本中的无依据确定性表述 |
| SectionReviewerAgent | 模型返回 `pass` 但同时提出问题 | 问题得到保留，结论确定性降级为 `revise` |
| ReportReviewerAgent | 全篇存在已知章节的范围问题 | 问题引用有效章节，结论降级为 `revise` |
| ChiefEditorAgent | 全篇存在口径限制且引用已稳定化 | 保留 Evidence 血缘、逐项处理问题并输出 `limited` |

“无依据生成”目前采用两层确定性防线：引用编号必须属于当前来源表；固定样本中的证据
限制必须保留，不能改写为确定性结论。真正的开放域事实正确性仍需要后续真实模型评测、
人工抽检或基于证据的语义评审，不能由字符串规则替代。

## 故障注入矩阵

`multi_agent_research/eval/faults.py` 可在模型、工具或生命周期事件持久化边界注入一次性
延迟或异常。注入器只供测试和离线评测使用，不接入生产工作流。

| 故障 | 注入位置 | 预期行为 |
|---|---|---|
| 模型超时 | Agent 模型调用 | 受 Agent 本地时限终止，记录 `agent_failed` |
| 检索超时 | 只读检索工具 | 记录工具失败和 Agent 暂停，不伪造空结果 |
| 工具中暂停 | 只读检索工具 | 原样传播 `ExecutionPaused`，保存暂停事件 |
| 预算耗尽 | 只读检索工具 | 原样传播 `BudgetExceeded`，等待明确追加预算 |
| 依赖失败 | 只读检索工具 | 安全包装异常，不把供应商消息写入 Trace |
| Checkpoint 后崩溃 | `agent_retrying` 已保存之后 | 无正常终态；新 `agent_run_id` 从局部状态继续 |
| 过期执行 | 生命周期持久化栅栏 | `StaleExecutionError` 立即停止，不再调用模型 |

恢复等价性比较业务结论、来源身份、检索轮次和内容；不要求恢复后新请求产生的
`retrieved_at` 与未中断执行逐字节一致，因为它们代表真实且不同的观测时间。

## 运行方式

```powershell
& 'D:\anaconda\envs\multi-agent\python.exe' -m pytest `
  tests\test_agent_behavior_eval.py tests\test_agent_fault_matrix.py -q
```

该命令只使用固定响应和内存数据。完整回归仍需运行全部 Python 测试、前端 DOM 测试、
Ruff 和 `git diff --check`。PostgreSQL 故障测试继续保持显式 opt-in，避免误连共享数据库。
