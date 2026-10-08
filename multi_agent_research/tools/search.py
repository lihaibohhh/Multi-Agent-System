from __future__ import annotations
import asyncio
from typing import Any
from langchain_core.tools import tool
from .support import _err, _ok, _shrink_search_results, with_retry
from ..core.config import settings
from ..core.budget import RunControlError, invoke_retrieval


# ================================================================================
# 工具 1：Web 搜索（Tavily API）
# ================================================================================
# 安装依赖：pip install tavily-python
# 配置环境变量：
#   - TAVILY_API_KEY：Tavily API 密钥（必须）
#   - MAX_SEARCH_RESULTS：最多返回多少条搜索结果（可选，默认 5）
@tool(
    description=(
        "【触发条件】以下情况调用本工具：\n"
        "  - 需要互联网上的最新动态、实时新闻、近期公告\n"
        "  - 需要验证公开事实（公司官网信息、开源版本号、政策发布日期等）\n"
        "  - query_internal_knowledge 已返回 has_relevant_content=False，"
        "且用户问题属于可公开查询的宏观/行业信息\n"
        "  示例 query：'2024年新能源补贴政策最新消息'、'比亚迪最新季报发布时间'\n\n"
        "【不触发条件】以下情况禁止调用本工具：\n"
        "  - 用户要求查询内部知识库中的具体财务数值或研报原文\n"
        "  - query_internal_knowledge 尚未调用（不得跳过直接搜索）\n"
        "  - 用户问题涉及私有文档中的具体数据，即使知识库未命中也不得用本工具补充\n\n"
        "【输入】简洁的搜索关键词或短句，不超过300字符\n"
        "  好的输入：'比亚迪 2024 Q3 业绩发布'\n"
        "  差的输入：'请帮我在网上搜索所有关于比亚迪的信息'\n\n"
        "【输出关键字段】\n"
        "  data.results[].title     — 结果标题\n"
        "  data.results[].url       — 来源链接，回答时应标注\n"
        "  data.results[].content   — 摘要，最多800字\n"
        "  data.results[].published_date — 发布日期，判断时效性用"
    )
)
@with_retry(
    tool_name="search",
    max_retries=settings.tools.search.max_retries,
    timeout=settings.tools.search.timeout
)
async def search(query: str) -> dict[str, Any]:
    """
    Web 搜索工具 — 通过 Tavily API 搜索互联网最新信息。

    功能描述：
      - 接收用户查询，通过 Tavily API 搜索互联网
      - 裁剪搜索结果以适应上下文大小
      - 返回前几条最相关的结果（标题、链接、摘要、发布时间）

    参数：
      query (str)：搜索关键词或短句
                   - 不能为空
                   - 最多 300 字符（超出会自动截断）
                   - 可以使用自然语言问题或关键词

    返回：
      {
        “ok”: True/False,
        “data”: {
          “results”: [                    # 搜索结果列表
            {
              “title”: str,               # 搜索结果标题
              “url”: str,                 # 链接
              “content”: str,             # 摘要（最多 800 字）
              “score”: float,             # 匹配分数（可能为 None）
              “published_date”: str       # 发布日期（可能为 None）
            },
            ...
          ],
          “answer”: str                   # Tavily 生成的简洁回答（可选）
        },
        “error”: {...}                    # 失败时的错误信息
      }

    执行逻辑：
      1. 输入清理：strip() 并检查是否为空
      2. 输入截断：超过 300 字符自动截断
      3. 依赖检查：延迟 import，缺少依赖返回友好错误
      4. 密钥检查：检查 TAVILY_API_KEY 环境变量
      5. 配置读取：从 runtime context 或 MAX_SEARCH_RESULTS 环境变量读取返回数
      6. API 调用：使用 TavilySearch.ainvoke() 执行搜索
      7. 结果裁剪：通过 _shrink_search_results() 压缩到适当大小
      8. 异常处理：临时错误（超时、连接）raise 给 with_retry，永久错误返回 _err()

    错误场景：
      - BAD_INPUT：query 为空
      - MISSING_DEPENDENCY：未安装 langchain-tavily
      - MISSING_API_KEY：缺少 TAVILY_API_KEY 环境变量
      - SEARCH_FAILED：API 调用出错（非临时错误）
      - 超时/连接错误：raise，由 with_retry 自动重试 2 次

    示例：
      result = await search(“Python 3.12 最新功能”)
      # 返回关于 Python 3.12 的最新搜索结果
    """
    tool_name = "search"
    q = (query or "").strip()

    # ════════════════════ 阶段 1：输入校验 ════════════════════
    if not q:
        return _err(
            tool_name=tool_name,
            query=q,
            code="BAD_INPUT",
            message="query 不能为空，请提供要搜索的关键词。"
        )
    if len(q) > 300:
        q = q[:300]

    # ════════════════════ 阶段 2：依赖检查 ════════════════════
    # 延迟 import 是为了：如果这个工具没有被使用，就不会强制安装依赖
    # 或者如果依赖缺失，整个 tools.py 模块也不会因此 import 失败
    try:
        from langchain_tavily import TavilySearch
    except ImportError:
        return _err(
            tool_name=tool_name,
            query=q,
            code="MISSING_DEPENDENCY",
            message="未安装依赖 langchain-tavily。请执行：pip install langchain-tavily",
        )

    # ════════════════════ 阶段 3：密钥检查 ════════════════════
    # Tavily API 需要密钥认证，提前检查能更快给出友好错误
    if not settings.tool_secrets.tavily_api_key.get_secret_value():
        return _err(
            tool_name=tool_name,
            query=q,
            code="MISSING_API_KEY",
            message="缺少环境变量 TAVILY_API_KEY，搜索工具不可用。请先设置该 Key。",
        )

    # ════════════════════ 阶段 4：执行搜索 ════════════════════
    try:
        # 读取配置：最多返回几条结果
        max_results = max(1, min(int(settings.tools.search.max_results), 20))

        # 初始化搜索器
        wrapped = TavilySearch(max_results=max_results)

        raw = await invoke_retrieval(lambda: wrapped.ainvoke({"query": q}), label="web_tool")
        if isinstance(raw, dict) and isinstance(raw.get("error"), Exception):
            raise raw["error"]

        # 裁剪结果以适应上下文
        data = _shrink_search_results(raw, max_items=min(5, max_results))
        return _ok(
            tool_name=tool_name,
            query=q,
            data=data,
            meta={"max_results": max_results},
        )

    # ════════════════════ 阶段 5：异常处理 ════════════════════
    except RunControlError:
        raise
    except (TimeoutError, asyncio.TimeoutError, ConnectionError, OSError):
        # 临时错误：raise 给 with_retry 装饰器，自动重试
        # 不能在这里 try-except 吞掉，否则 with_retry 无法捕获
        raise
    except Exception as e:
        # 永久错误（API 返回 4xx、解析失败等）：返回失败结果，不重试
        return _err(
            tool_name=tool_name,
            query=q,
            code="SEARCH_FAILED",
            message=f"搜索工具调用失败：{type(e).__name__}: {e}",
        )
