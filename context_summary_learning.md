# LLM Agent 对话历史压缩：经验文档

> 基于 DeerFlow 项目 `SummarizationMiddleware` 的实践总结。
> 适用场景：任何基于 LangChain / LangGraph 构建的长对话 Agent。

---

## 一、为什么需要压缩

LLM 的上下文窗口有硬性 token 上限。对话越长，消耗越多，超限后调用直接报错。
简单的解法是"截断历史"，但会让 Agent 失忆。更好的解法是**用 LLM 把旧历史压缩成摘要**，既控制长度，又保留语义。

```
原始历史（634 tok）→ 摘要（~60 tok）+ 近期消息（~170 tok）= 230 tok  ✓
```

---

## 二、核心流程（六步）

```
消息列表超出 token 阈值
        ↓
① 确定截断点（cutoff_index）
        ↓
② 分区：待摘要 vs 保留
        ↓
③ 两层特殊保护（Skill 抢救 + 上下文提醒保护）
        ↓
④ 构建 Prompt → 调用 LLM → 得到摘要字符串
        ↓
⑤ 构建新消息列表
        ↓
⑥ 写回状态（RemoveAll + 新列表）
```

---

## 三、截断点的确定

从末尾向前数，保留最近 N 条消息（`keep_last_n_messages`），其余全部进入待摘要区。

**关键约束**：截断点不能落在工具调用对的中间。

```
❌  错误切法：
    [AI: 调用 read_file]  ← cutoff
    [Tool: 文件内容]      ← 保留
    → ToolMessage 失去对应的 AIMessage，消息链断裂

✓  正确切法：
    [AI: 调用 read_file]  ← 自动向后移动
    [Tool: 文件内容]      ← 一起保留
    ─── cutoff ───
```

触发条件：`total_tokens > max_tokens` 且 `len(messages) > keep_last_n_messages`。

---

## 四、Skill 抢救机制

### 问题

Agent 在执行任务前会通过 `read_file` 读取技能文件（如 `SKILL.md`），返回的内容可达数千 token。若这段交互被压缩进摘要，模型就会"忘记"技能内容，质量大幅退化。

### 解法：识别 → 预算筛选 → 移出摘要区

**Step 1：识别 Skill Bundle**

扫描待摘要区，找出满足以下条件的消息对：

- AIMessage 中包含指向 `/mnt/skills/` 路径的工具调用（`read_file`、`view`、`cat` 等）
- 紧随其后的 ToolMessage 是该调用的返回结果

```python
# 识别标准
is_skill_call = (
    tool_call["name"] in {"read_file", "view", "cat"}
    and tool_call["args"]["path"].startswith("/mnt/skills/")
)
```

每对 AI + Tool 消息组成一个 `SkillBundle`，记录其 token 数和文件路径（作为去重 key）。

**Step 2：贪心预算筛选（从最新开始）**

```
三重约束，同时满足才保留：
  - 单个 Bundle token 数 ≤ per_skill_limit（默认 5000）
  - 累计 token 数 ≤ total_skill_limit（默认 25000）
  - 保留数量 ≤ max_skill_count（默认 5）
  - 相同文件路径只保留一次（去重）
```

**Step 3：消息拆分**

一条 AIMessage 可能同时含有 skill 调用和普通调用，需要克隆拆分：

```python
# 拆成两条 AIMessage
rescued_part  = clone(ai_msg, tool_calls=skill_calls,   content="")
remaining_part = clone(ai_msg, tool_calls=other_calls)

# rescued_part  → 移入保留区
# remaining_part → 留在待摘要区
```

---

## 五、上下文提醒保护机制

### 问题

`DynamicContextMiddleware` 会在对话开头注入一条隐藏消息（携带当前日期、用户记忆等），作为上下文锚点。若被摘要压缩删除，该中间件下次运行会找错位置重新注入，导致消息顺序混乱。

### 解法：直接移出，放到保留区最前面

```python
reminders = [m for m in to_summarize if is_dynamic_context_reminder(m)]
remaining  = [m for m in to_summarize if not is_dynamic_context_reminder(m)]

# 保留区最终顺序：reminder → skill bundles → 近期消息
return remaining, reminders + preserved_messages
```

**顺序很重要**：reminder 必须排在保留区最前，才能让 DynamicContextMiddleware 正确识别"提醒已存在，无需重新注入"。

---

## 六、Prompt 构建与 LLM 调用

### 消息序列化

用 `get_buffer_string()` 而不是 `str(message)`：

```python
# ✓ 正确：只保留角色和内容，不含 id / metadata
formatted = get_buffer_string(messages)
# → "Human: 帮我分析财报\nAI: 营收同比+12%\n..."

# ✗ 错误：str(message) 会包含大量无用字段，虚增 token
```

### Prompt 模板

```
Progressively summarize the lines of conversation provided,
adding onto the previous summary returning a new summary.

New lines of conversation:
{messages}

New summary:
```

LLM 续写 `New summary:` 后的内容即为压缩结果。

### 防止摘要 token 流泄露到前端

摘要调用使用**独立的模型副本**，不修改原始 `self.model`：

```python
# 构造函数中，创建一次，全程复用
self._summary_model = self.model.with_config(tags=[*existing_tags, TAG_NOSTREAM])
```

`TAG_NOSTREAM` 让前端的 stream callback 跳过这次调用，用户完全无感知。

> **为什么不能直接改 `self.model`？**
> 中间件实例在并发协程间共享。临时修改 `self.model`，在 `await` 挂起期间会被其他协程读到污染状态。`with_config()` 创建副本，是并发安全的唯一正确解法。

### 降级处理

```python
try:
    response = self._summary_model.invoke(prompt, ...)
    return response.text.strip()
except Exception as e:
    return f"Error generating summary: {e!s}"   # 失败不崩溃，返回报错描述
```

---

## 七、摘要消息的写回

### 摘要消息格式

```python
HumanMessage(
    content="Here is a summary of the conversation to date:\n\n{summary}",
    name="summary"    # 特殊标记，前端识别后隐藏，不展示给用户
)
```

### 最终消息列表结构

```
[HumanMessage(name="summary")]   ← LLM 可见，用户不可见
[AI + Tool（rescued skill）]      ← Skill 保留区
[AI + Tool（rescued skill）]
[最近 N 条普通消息]               ← 原始保留区
```

### 状态写回机制（LangGraph）

```python
return {
    "messages": [
        RemoveMessage(id=REMOVE_ALL_MESSAGES),  # 清空所有现有消息
        *new_messages,                          # 摘要消息
        *preserved_messages,                    # skill bundles + 近期消息
    ]
}
```

LangGraph 的 `add_messages` reducer 处理这个字典：先执行清空，再逐条追加，整个过程对 Agent 透明。

```python
# State 定义中的 reducer 声明
class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    #                                     ↑ 这个函数决定如何合并更新
```

---

## 八、可复用的设计模式

| 模式 | 应用场景 | 本项目体现 |
|---|---|---|
| 不可变副本 | 并发安全地修改配置 | `model.with_config()` 创建 `_summary_model` |
| 贪心预算 | 在多重约束下最大化保留 | `_select_bundles_to_rescue` |
| 状态指令包 | 让中间件与状态解耦 | 返回 `{"messages": [...]}` 而非直接操作列表 |
| 降级不崩溃 | LLM 调用失败时保持系统可用 | `try/except` 返回错误描述字符串 |
| 钩子系统 | 让摘要事件可被外部监听 | `BeforeSummarizationHook` + `_fire_hooks` |
| 中间件隔离 | 防止中间件之间隐性耦合 | `_preserve_dynamic_context_reminders` |

---

## 九、关键参数速查

| 参数 | 默认值 | 含义 |
|---|---|---|
| `max_tokens` | 由父类配置 | 超过此值触发摘要 |
| `keep_last_n_messages` | 由父类配置 | 摘要后保留的最近消息数 |
| `preserve_recent_skill_count` | 5 | 最多抢救的 Skill Bundle 数量 |
| `preserve_recent_skill_tokens` | 25000 | 抢救 Skill 的总 token 预算 |
| `preserve_recent_skill_tokens_per_skill` | 5000 | 单个 Skill Bundle 的 token 上限 |
| `skills_container_path` | `/mnt/skills` | Skill 文件的根路径 |
| `skill_file_read_tool_names` | `{"read_file", "read", "view", "cat"}` | 识别为 Skill 读取的工具名 |