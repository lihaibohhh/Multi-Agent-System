# =============================================================================
# config.py — Multi-Agent Project Configuration
# Framework: LangChain / LangGraph
# Settings:  pydantic-settings v2
#
# 依赖安装：
#   pip install "pydantic-settings>=2.0" "pydantic>=2.0" python-dotenv
# =============================================================================

from __future__ import annotations

import logging
import os
import yaml
from enum import Enum
from functools import lru_cache
from pathlib import Path
from typing import List, Literal

from dotenv import load_dotenv
from pydantic import AliasChoices, Field, SecretStr, field_validator, model_validator, BaseModel
from pydantic_settings import BaseSettings, SettingsConfigDict


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 环境标识
# ---------------------------------------------------------------------------
class Environment(str, Enum):
    DEV = "development"
    TEST = "testing"
    PROD = "production"


def _resolve_project_root() -> Path:
    """
    解析项目根目录。

    优先级：
    1. 环境变量 PROJECT_ROOT（显式指定，跨机器/跨平台最稳妥，多人协作或
       部署到 Docker 时建议设置）；
    2. 没有设置时，退化为当前文件所在目录，仅作为本地开发兜底，
       不依赖任何与具体开发机相关的假设（不再像早期版本那样把某个人
       电脑上的绝对路径写进注释）。
    """
    env_root = os.getenv("PROJECT_ROOT")
    if env_root:
        return Path(env_root).resolve()
    return Path(__file__).resolve().parent.parent.parent


PROJECT_ROOT: Path = _resolve_project_root()
_ENV_PATH: Path = PROJECT_ROOT / ".env"
_YAML_PATH: Path = PROJECT_ROOT / "config.yaml"

load_dotenv(dotenv_path=_ENV_PATH, override=False)


# =============================================================================
# 1. LLM Provider 配置（env 层）
#    OpenAI / Anthropic / Deepseek / Azure 共用字段较多（api_key / base_url /
#    model / temperature / max_tokens / timeout / retries），抽出基类避免
#    四份几乎相同的字段定义。
# =============================================================================
class LLMProviderConfig(BaseSettings):
    """所有 LLM Provider 配置的公共基类，不直接实例化。"""

    model_config = SettingsConfigDict(populate_by_name=True, extra="ignore")

    api_key: SecretStr = Field(default="")
    base_url: str = Field(default="")
    model: str = Field(default="")
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    max_tokens: int = Field(default=4096, gt=0)
    timeout: int = Field(default=60, gt=0)
    retries: int = Field(default=2, ge=0)

    @property
    def is_configured(self) -> bool:
        """是否已提供 API Key，用于运行期判断该 Provider 是否可用。"""
        return bool(self.api_key.get_secret_value())


class OpenAIConfig(LLMProviderConfig):
    """env_prefix="OPENAI_" → 读取 OPENAI_API_KEY / OPENAI_MODEL 等"""

    model_config = SettingsConfigDict(
        env_prefix="OPENAI_",
        populate_by_name=True,  # 允许在测试中直接按字段名赋值
        extra="ignore"
    )

    model: str = Field(default="gpt-4o")


class AnthropicConfig(LLMProviderConfig):
    """env_prefix="ANTHROPIC_" → 读取 ANTHROPIC_API_KEY / ANTHROPIC_MODEL"""

    model_config = SettingsConfigDict(
        env_prefix="ANTHROPIC_",
        populate_by_name=True,
        extra="ignore"
    )

    model: str = Field(default="claude-sonnet-4-20250514")


class DeepseekConfig(LLMProviderConfig):
    """env_prefix=DEEPSEEK_ -> 读取 DEEPSEEK_API_KEY / DEEPSEEK_MODEL 等"""
    model_config = SettingsConfigDict(
        env_prefix="DEEPSEEK_",
        populate_by_name=True,
        extra="ignore"
    )

    base_url: str = Field(default="https://api.deepseek.com")
    model:    str = Field(default="deepseek-chat")


class AzureConfig(LLMProviderConfig):
    """Azure OpenAI — 可选，三个字段都为空时视为未启用。"""

    model_config = SettingsConfigDict(
        env_prefix="AZURE_OPENAI_",
        populate_by_name=True,
        extra="ignore"
    )

    base_url:    str = Field(default="http://localhost:8080/v1", description="AZURE_OPENAI_BASE_URL")
    endpoint: str = Field(default="")
    # 历史命名与 prefix 不一致的字段，用 validation_alias 精确映射
    deployment: str = Field(default="", validation_alias="AZURE_DEPLOYMENT_NAME")
    api_version: str = Field(default="2024-02-01", validation_alias="AZURE_API_VERSION")
    max_tokens: int = Field(default=2048, gt=0)


    @property
    def enabled(self) -> bool:
        return bool(self.api_key.get_secret_value() and self.endpoint and self.deployment)


class LocalOpenAIConfig(LLMProviderConfig):
    """Local OpenAI-compatible endpoint used by vLLM or a similar server."""

    model_config = SettingsConfigDict(
        env_prefix="LOCAL_LLM_",
        populate_by_name=True,
        extra="ignore",
    )

    api_key: SecretStr = Field(default="local")
    base_url: str = Field(default="http://localhost:8000/v1")
    model: str = Field(default="Qwen/Qwen2.5-7B-Instruct-GPTQ-Int4")


# ---------------------------------------------------------------------------
# 3. 隔壁知识库服务（仅调用检索与健康检查接口）
# ---------------------------------------------------------------------------
class KnowledgeServiceConfig(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="KNOWLEDGE_SERVICE_",
        populate_by_name=True,
        extra="ignore",
    )

    base_url: str = Field(default="http://127.0.0.1:8001")
    api_key: SecretStr = Field(default="")
    timeout: float = Field(default=60.0, gt=0)
    top_k: int = Field(default=3, ge=1, le=10)
    retrieval_mode: Literal["hybrid", "bm25", "vector"] = "hybrid"


# ---------------------------------------------------------------------------
# 4. 数据库 / 缓存
# ---------------------------------------------------------------------------
class DatabaseConfig(BaseSettings):
    """
    env_prefix="DB_" → DB_POOL_SIZE / DB_MAX_OVERFLOW
    其余字段因历史命名与 prefix 不符，通过 validation_alias 精确映射。
    """

    model_config = SettingsConfigDict(
        env_prefix="DB_",
        populate_by_name=True,
        extra="ignore",
    )

    # 关系型数据库。POSTGRES_DB_URL 是本项目主名称，DATABASE_URL 保持向后兼容。
    url:          str = Field(
        default="postgresql://user:password@localhost:5432/mydb",
        validation_alias=AliasChoices("POSTGRES_DB_URL", "DATABASE_URL"),
    )
    pool_size:    int = Field(default=5, gt=0, description="DB_POOL_SIZE")
    max_overflow: int = Field(default=10, ge=0, description="DB_MAX_OVERFLOW")


class RedisConfig(BaseSettings):
    """
    env_prefix="REDIS_" → 自动读取 REDIS_URL / REDIS_MAX_CONNECTIONS 等
    原版用 @model_validator 手动循环 os.getenv，此处用 BaseSettings 机制统一处理。
    """

    model_config = SettingsConfigDict(
        env_prefix="REDIS_",
        populate_by_name=True,
        extra="ignore",
    )

    url:                str = Field(default="redis://localhost:6379", description="REDIS_URL")
    max_connections:    int = Field(default=20, gt=0, description="REDIS_MAX_CONNECTIONS")

# ---------------------------------------------------------------------------
# 6. 工具
# ---------------------------------------------------------------------------
class ToolConfig(BaseSettings):
    model_config = SettingsConfigDict(
        populate_by_name=True,
        extra="ignore",
    )

    tavily_api_key:  SecretStr = Field(default="", validation_alias="TAVILY_API_KEY")
    serpapi_key:     SecretStr = Field(default="", validation_alias="SERPAPI_API_KEY")

    # 部署级开关（不同环境行为不同，适合放 env）
    enable_code_exec: bool = Field(default=False, validation_alias="ENABLE_CODE_EXEC")


# ---------------------------------------------------------------------------
# 7. Agent / LangGraph
# ---------------------------------------------------------------------------
class AgentConfig(BaseSettings):
    """
    env_prefix="AGENT_" → AGENT_MAX_ITERATIONS 等
    LangGraph 相关变量命名为 LANGGRAPH_*，通过 validation_alias 映射。
    """

    model_config = SettingsConfigDict(
        env_prefix="AGENT_",
        populate_by_name=True,
        extra="ignore",
    )

    max_iterations:       int = Field(default=10, gt=0, description="AGENT_MAX_ITERATIONS")
    max_retries:          int = Field(default=3, ge=0, description="AGENT_MAX_RETRIES")
    retry_delay:          float = Field(default=1.0, ge=0, description="AGENT_RETRY_DELAY")
    stream:               bool = Field(default=True, description="AGENT_STREAM")

    section_model: str = "deepseek/deepseek-chat"
    section_max_count: int = Field(default=4, ge=1, le=4)
    section_max_search_rounds: int = Field(default=2, ge=1, le=4)
    section_max_revisions: int = Field(default=1, ge=0, le=3)

    run_max_model_calls: int = Field(default=80, gt=0, le=1000)
    run_max_tokens: int = Field(default=800_000, gt=0)
    run_max_retrieval_calls: int = Field(default=80, gt=0, le=1000)
    run_timeout: float = Field(default=3600, gt=0, description="每次执行的时限；恢复重新计时，累计调用额度不重置")
    model_call_timeout: float = Field(default=120, gt=0)
    retrieval_call_timeout: float = Field(default=60, gt=0)
    retrieval_concurrency: int = Field(default=4, ge=1, le=32)

    enable_checkpointing: bool = Field(default=True, validation_alias="LANGGRAPH_CHECKPOINTING")
    recursion_limit:      int = Field(default=25, gt=0, validation_alias="LANGGRAPH_RECURSION_LIMIT")
    checkpoint_backend:   str = Field(default="memory", validation_alias="CHECKPOINT_BACKEND",
                                    description="memory | sqlite | postgres | redis | none")
    checkpoint_db_path:   str = Field(default="./checkpoints/research.sqlite3",
                                    validation_alias="CHECKPOINT_DB_PATH")
    timezone:             str = "Asia/Shanghai"


# ---------------------------------------------------------------------------
# 8. 日志 / LangSmith
# ---------------------------------------------------------------------------
class LoggingConfig(BaseSettings):
    """
    env_prefix="LOG_" → LOG_LEVEL / LOG_FORMAT / LOG_FILE
    LangChain/LangSmith 相关变量通过 validation_alias 映射。
    """

    model_config = SettingsConfigDict(
        env_prefix="LOG_",
        populate_by_name=True,
        extra="ignore",
    )

    level:             str = Field(default="INFO", description="LOG_LEVEL")
    format:            str = Field(
        default="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        description="LOG_FORMAT",
    )
    file:              str = Field(default="", description="LOG_FILE  # 空 = 仅控制台")
    langchain_verbose: bool = Field(default=False, validation_alias="LANGCHAIN_VERBOSE")
    langsmith_enabled: bool = Field(default=False, validation_alias="LANGCHAIN_TRACING_V2")
    langsmith_api_key: SecretStr = Field(default="", validation_alias="LANGCHAIN_API_KEY")
    langsmith_project: str = Field(default="multi-agent", validation_alias="LANGCHAIN_PROJECT")


    def setup(self) -> None:
        """初始化 logging；由 Settings._bootstrap() 在启动时调用。"""
        handlers: list[logging.Handler] = [logging.StreamHandler()]
        if self.file:
            handlers.append(logging.FileHandler(self.file, encoding="utf-8"))

        logging.basicConfig(
            level=getattr(logging, self.level, logging.INFO),
            format=self.format,
            handlers=handlers,
            force=True,  # 防止多次 basicConfig 调用无效
        )

        if self.langchain_verbose:
            import langchain
            langchain.verbose = True

        if self.langsmith_enabled:
            key = self.langsmith_api_key.get_secret_value()
            if key:
                os.environ.update({
                    "LANGCHAIN_TRACING_V2": "true",
                    "LANGCHAIN_API_KEY": key,
                    "LANGCHAIN_PROJECT": self.langsmith_project,
                })


# ===========================================================================
# 第二层：YAML 层（BaseModel，不读 env）
# 职责：工具参数、路径、超时——适合提交 git 让团队共享的内容
# ===========================================================================
class KnowledgeYamlConfig(BaseModel):
    max_content_chars: int = 800
    max_retries: int = 2
    timeout: int = 30


class SearchYamlConfig(BaseModel):
    max_results: int = 5
    timeout:     int = 15
    max_retries: int = 2


class CodeExecYamlConfig(BaseModel):
    timeout:            int = 30
    workspace_dir:      str = str(PROJECT_ROOT / "workspace")
    allowed_extensions: List[str] = [".txt", ".csv", ".json", ".pdf"]

    @field_validator("allowed_extensions", mode="before")
    @classmethod
    def _parse_extensions(cls, v: str | list) -> list[str]:
        """兼容 yaml 里写成字符串的情况：.txt,.csv,.json"""
        if isinstance(v, str):
            return [ext.strip() for ext in v.split(",") if ext.strip()]
        return v


class ToolsYamlConfig(BaseModel):
    knowledge: KnowledgeYamlConfig = Field(default_factory=KnowledgeYamlConfig)
    search: SearchYamlConfig = Field(default_factory=SearchYamlConfig)
    code_exec: CodeExecYamlConfig = Field(default_factory=CodeExecYamlConfig)


class YamlConfig(BaseModel):
    tools: ToolsYamlConfig = Field(default_factory=ToolsYamlConfig)


def _load_yaml() -> YamlConfig:
    """加载 config.yaml；文件不存在时静默使用代码默认值。"""
    if _YAML_PATH.exists():
        with open(_YAML_PATH, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        return YamlConfig(**data)
    logging.getLogger(__name__).warning(
        "未找到 config.yaml（%s），使用代码默认值", _YAML_PATH
    )
    return YamlConfig()


# ===========================================================================
# 第三层：Settings 聚合器
# ===========================================================================
class Settings(BaseSettings):
    """
    项目全局配置单例。
    各子配置类独立从环境变量读取（load_dotenv 已在模块顶部加载 .env）。
    """

    model_config = SettingsConfigDict(
        populate_by_name=True,
        extra="ignore",  # 忽略 .env 中多余的变量，不抛错
    )

    app_env: Environment = Field(
        default=Environment.DEV,
        validation_alias="APP_ENV",
        description="APP_ENV  # development | testing | production",
    )

    # env 层子配置
    openai:       OpenAIConfig = Field(default_factory=OpenAIConfig)
    anthropic:    AnthropicConfig = Field(default_factory=AnthropicConfig)
    deepseek:     DeepseekConfig = Field(default_factory=DeepseekConfig)
    azure:        AzureConfig = Field(default_factory=AzureConfig)
    local:        LocalOpenAIConfig = Field(default_factory=LocalOpenAIConfig)
    database:     DatabaseConfig = Field(default_factory=DatabaseConfig)
    redis:        RedisConfig = Field(default_factory=RedisConfig)
    knowledge_service: KnowledgeServiceConfig = Field(default_factory=KnowledgeServiceConfig)
    tool_secrets: ToolConfig = Field(default_factory=ToolConfig)
    agent:        AgentConfig = Field(default_factory=AgentConfig)
    logging:      LoggingConfig = Field(default_factory=LoggingConfig)

    # YAML 层
    yaml: YamlConfig = Field(default_factory=_load_yaml)

    @model_validator(mode="after")
    def _validate_has_llm_key(self) -> "Settings":
        """纯数据校验：至少要有一个可用的 LLM Provider。不在此处做日志/IO。"""
        has_any_key = any(
            p.is_configured for p in (self.openai, self.anthropic, self.deepseek)
        ) or self.azure.enabled
        if not has_any_key:
            raise ValueError(
                "至少需要配置一个 LLM Provider 的 API Key：\n"
                "  OPENAI_API_KEY / ANTHROPIC_API_KEY / DEEPSEEK_API_KEY / "
                "(AZURE_OPENAI_API_KEY + AZURE_OPENAI_ENDPOINT + AZURE_DEPLOYMENT_NAME)"
            )
        return self

    # ------------------------------------------------------------------
    # 便捷属性：在其他模块里按需使用
    # ------------------------------------------------------------------
    @property
    def is_dev(self) -> bool:
        return self.app_env == Environment.DEV

    @property
    def is_test(self) -> bool:
        return self.app_env == Environment.TEST

    @property
    def is_prod(self) -> bool:
        return self.app_env == Environment.PROD

    @property
    def tools(self) -> ToolsYamlConfig:
        """
        返回 YAML 层的工具配置。
        """
        return self.yaml.tools


# ---------------------------------------------------------------------------
# 全局单例（其他模块直接 from config import settings）
# ---------------------------------------------------------------------------
@lru_cache
def get_settings() -> Settings:
    """
    获取全局配置单例（进程内只构造一次，lru_cache 缓存）。

    测试中可用 get_settings.cache_clear() 配合 monkeypatch 环境变量重新加载。
    """
    return Settings()


settings = get_settings()


# ---------------------------------------------------------------------------
# 快速自检（python config.py）
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import pprint
    # mode="json" 将 SecretStr 序列化为 "**********"，不泄露密钥
    pprint.pprint(settings.model_dump(mode="json"))
