# 人工冒烟测试清单

本清单是旧链路清理后的人工验收闸门。自动化回归通过并不等价于真实模型、
knowledge-service、浏览器 SSE 和持久化恢复均可用。完成本清单并确认结果前，不进入
章节 DAG 并行改造。

## 前置条件

- 使用 `multi-agent` 虚拟环境；以下命令固定使用
  `D:\anaconda\envs\multi-agent\python.exe`。
- `.env` 已配置至少一个模型 Provider；需要联网来源时再配置 `TAVILY_API_KEY`。
- 隔壁 knowledge-service 已由其所属项目启动。本项目不得直接访问其 Chroma 目录，
  也不得调用任何写入、管理或缓存失效接口。
- 使用测试数据或新的 Run；不要恢复 v4 及更早的在途任务。

## 1. 验证只读检索边界

```powershell
& 'D:\anaconda\envs\multi-agent\python.exe' scripts\verify_knowledge_service.py '人工智能 教育'
```

预期：显示 `ready=True`，并返回检索条数。客户端固定发送
`use_query_cache=false`；该脚本只做健康检查和检索。

## 2. 启动本项目

```powershell
& 'D:\anaconda\envs\multi-agent\python.exe' -m uvicorn multi_agent_research.api.server:app --host 127.0.0.1 --port 8000 --loop multi_agent_research.api.event_loop:selector_loop_factory
```

另开 PowerShell 检查：

```powershell
Invoke-RestMethod http://127.0.0.1:8000/api/health
```

预期：HTTP 200；业务仓储、Checkpoint 后端、实例所有权和后台监视器均处于可用状态。

## 3. 浏览器主流程

打开 `http://127.0.0.1:8000`，创建一个范围明确、可在少量章节内完成的新研究任务。

逐项确认：

- 任务可以创建并启动，SSE 持续显示进度，刷新页面后仍能恢复当前 Run。
- 执行角色只体现 Planner、EvidenceResearch、SectionWriter、SectionReviewer、
  ReportReviewer、ChiefEditor；Claim 绑定体现为 Processor，不出现旧 Supervisor、
  SearchAgent、EvidenceAnalyst 或 ClaimExtractor 链路。
- 每一章按“研究、写作、审校、Claim 绑定”推进，完成后再进入下一章。
- 全章初审后进入 ChiefEditor 跨章编辑，再由 ReportReviewer 做编辑后复审；最终报告
  能装配完成，章节引用编号有效，来源卡片和限制项可见。
- Agent Trace 只显示安全元数据，不泄露 Prompt、检索正文或本地 Checkpoint 内容。

## 4. 暂停与恢复

新建或使用尚未完成的测试 Run：

1. 执行中点击暂停，等待状态真正变成“已暂停”。
2. 刷新页面，确认章节成果和进度仍在。
3. 点击继续，确认从持久化边界恢复，而不是新建另一条旧版工作流。
4. 完成后确认没有重复章节、重复来源或用量倒退。

## 5. 验收记录

记录测试时间、模型 Provider、Checkpoint 后端、是否启用 Tavily，以及失败时的 Run ID。
不要记录密钥、Prompt 全文或检索正文。若以上任一关键项失败，先修复串行基线；只有
全部通过并由用户明确确认后，才进入 DAG 并行评估。
