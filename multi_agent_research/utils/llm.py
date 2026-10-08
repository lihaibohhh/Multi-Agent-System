# SNAPSHOT: (2026-01-31-14:26) → 修改于 config 统一化重构
"""
multi_agent_research/utils/llm.py

职责：
- 解析 Context.model（"provider/model-name"）
- 根据 provider 加载对应的 Chat Model
- 推理参数统一从 settings 读取，不再散写 os.getenv()

修改说明：
- ChatModelSettings 改为从 settings.llm 读取，移除 os.getenv 直接调用
- Provider credentials and endpoints are read from the matching settings section.
- 对外接口 load_chat_model() 不变，graph.py 无需任何修改
"""

from __future__ import annotations

from functools import lru_cache
from typing import Tuple

from langchain_core.language_models.chat_models import BaseChatModel

# ✅ 唯一改动：从 config 读，不再 import os + os.getenv
from ..core.config import settings


def _parse_model_ref(model_ref: str) -> Tuple[str, str]:
    """
    解析形如 "provider/model-name" 的字符串。
    """
    model_ref = (model_ref or "").strip()
    if not model_ref:
        raise ValueError("Context.model 不能为空，例如：'openai/gpt-4.1-mini'")

    if "/" not in model_ref:
        raise ValueError(
            "Context.model 必须是 'provider/model-name' 格式，例如："
            "'anthropic/claude-sonnet-4-5' 或 'openai/gpt-4.1-mini' 或 'local/qwen2.5-7b'"
        )
    provider, model_name = model_ref.split("/", 1)
    provider = provider.strip().lower()
    model_name = model_name.strip()
    if not provider or not model_name:
        raise ValueError(
            "Context.model 格式不正确，应该是 'provider/model-name'，例如 'openai/gpt-4.1-mini'"
        )
    return provider, model_name


# -----------------------------
# provider -> ChatModel 工厂
# -----------------------------

def _build_openai(model_name: str) -> BaseChatModel:
    try:
        from langchain_openai import ChatOpenAI
    except ImportError as e:
        raise ImportError("未安装依赖：pip install langchain-openai") from e
    cfg = settings.openai
    return ChatOpenAI(
        model=model_name,
        temperature=cfg.temperature,
        max_tokens=cfg.max_tokens,
        timeout=cfg.timeout,
        max_retries=0,  # Each external attempt must pass the Run budget gate.
        # langchain_openai 会自动读取 OPENAI_API_KEY 环境变量
        # load_dotenv 已在 config.py 里写入 os.environ，此处无需显式传递
    )


def _build_anthropic(model_name: str) -> BaseChatModel:
    try:
        from langchain_anthropic import ChatAnthropic
    except ImportError as e:
        raise ImportError("未安装依赖：pip install langchain-anthropic") from e
    cfg = settings.anthropic
    return ChatAnthropic(
        model=model_name,
        temperature=cfg.temperature,
        max_tokens=cfg.max_tokens,
        timeout=cfg.timeout,
        max_retries=0,
    )


def _build_local_openai_compatible(model_name: str) -> BaseChatModel:
    try:
        from langchain_openai import ChatOpenAI
    except ImportError as e:
        raise ImportError("未安装依赖：pip install langchain-openai") from e
    cfg = settings.local
    base_url = cfg.base_url
    api_key = cfg.api_key.get_secret_value() or "local"
    return ChatOpenAI(
        model=model_name,
        base_url=base_url,
        api_key=api_key,
        temperature=cfg.temperature,
        max_tokens=cfg.max_tokens,
        timeout=cfg.timeout,
        max_retries=0,
    )


def _build_deepseek_openai_compatible(model_name: str) -> BaseChatModel:
    try:
        from langchain_openai import ChatOpenAI
    except ImportError as e:
        raise ImportError("未安装依赖：pip install langchain-openai") from e
    cfg = settings.deepseek
    base_url = cfg.base_url
    api_key = cfg.api_key
    return ChatOpenAI(
        model=model_name,
        base_url=base_url,
        api_key=api_key,
        temperature=cfg.temperature,
        max_tokens=cfg.max_tokens,
        timeout=cfg.timeout,
        max_retries=0,
    )


# -----------------------------
# 对外唯一入口：load_chat_model
# -----------------------------

@lru_cache(maxsize=32)
def load_chat_model(model_ref: str) -> BaseChatModel:
    """
    graph.py 唯一依赖的函数，签名不变。
    """
    provider, model_name = _parse_model_ref(model_ref)

    if provider == "openai":
        return _build_openai(model_name)

    if provider == "anthropic":
        return _build_anthropic(model_name)

    if provider in ("local", "qwen-local", "openai-compatible"):
        return _build_local_openai_compatible(model_name)

    if provider in ("deepseek", "ds"):
        return _build_deepseek_openai_compatible(model_name)

    raise ValueError(
        f"不支持的 provider：{provider}。支持：openai / anthropic / local / deepseek\n"
        f"你传入的是：{model_ref}"
    )
