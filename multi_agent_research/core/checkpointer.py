"""
checkpointer.py — LangGraph Checkpointer 工厂（多 Agent 版）

从 multi_agent_research.core.config.settings 直接读取后端配置，无需外部传入 Context 对象。
其余五大设计原则与单 Agent 版完全一致：
  1. 懒初始化锁        — asyncio.Lock 在首次调用时绑定当前 loop
  2. 精确 cache key    — 含连接参数，锁内只做 dict.get，无迭代竞态
  3. 统一降级          — 所有 backend 失败共享同一 MemorySaver 实例
  4. 生命周期解耦      — 外部资源由 _lifecycle 注册表统一管理
  5. 双 key 缓存       — 降级后同时写原始 key，防止重复尝试已知失败的 backend

配置来源（.env / 环境变量）：
  CHECKPOINT_BACKEND   — memory | sqlite | postgres | redis | none  (默认 memory)
  CHECKPOINT_DB_PATH   — SQLite 文件路径  (默认 ./checkpoints/research.sqlite3)
  POSTGRES_DB_URL      — Postgres 连接串（DATABASE_URL 仍兼容）
  REDIS_URL            — Redis 连接串     (已有，由 RedisConfig 读取)
"""

from __future__ import annotations

import asyncio
import logging
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from langgraph.checkpoint.base import BaseCheckpointSaver


# psycopg's async implementation requires a selector loop on Windows.
# This module is imported while the application is being assembled, before
# uvicorn/asyncio creates the serving loop.
if sys.platform == "win32" and hasattr(asyncio, "WindowsSelectorEventLoopPolicy"):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

try:
    from psycopg_pool import AsyncConnectionPool
    from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
except ImportError:
    AsyncConnectionPool = None  # type: ignore[assignment]
    AsyncPostgresSaver = None   # type: ignore[assignment]


class CheckpointerFactory:
    """Checkpointer 工厂（多 Agent 版）。"""

    # ── 类级缓存 ─────────────────────────────────────────────────────────────
    _instances:  Dict[str, BaseCheckpointSaver] = {}
    _lifecycle:  Dict[str, Dict[str, Any]]      = {}
    _lock:       Optional[asyncio.Lock]          = None
    _effective: Dict[str, str] = {}
    _logger =    logging.getLogger(__name__)

    _DEFAULT_SQLITE_PATH = "./checkpoints/research.sqlite3"

    # ── 内部工具 ──────────────────────────────────────────────────────────────

    @classmethod
    def _get_lock(cls) -> asyncio.Lock:
        """懒创建锁，绑定到当前 event loop（兼容 FastAPI 热重载 / asyncio.run 重入）。"""
        if cls._lock is None:
            cls._lock = asyncio.Lock()
        return cls._lock

    @classmethod
    def _log(cls, msg: str, level: str = "info") -> None:
        getattr(cls._logger, level, cls._logger.info)(msg)

    @classmethod
    def _normalize_backend(cls, backend: str) -> str:
        b = (backend or "").strip().lower()
        aliases: Dict[str, str] = {
            "":          "memory",
            "mem":       "memory",
            "memory":    "memory",
            "sqlite":    "sqlite",
            "sqlite3":   "sqlite",
            "aiosqlite": "sqlite",
            "postgres":  "postgres",
            "postgresql":"postgres",
            "pg":        "postgres",
            "redis":     "redis",
            "none":      "none",
            "off":       "none",
            "false":     "none",
            "0":         "none",
        }
        return aliases.get(b, b)

    # ── 配置读取（延迟导入，避免循环依赖 / settings 尚未初始化）────────────

    @classmethod
    def _sqlite_db_path(cls) -> Path:
        from multi_agent_research.core.config import settings
        raw = settings.agent.checkpoint_db_path or cls._DEFAULT_SQLITE_PATH
        return Path(raw).expanduser().resolve()

    @classmethod
    def _postgres_conn_str(cls) -> str:
        """读取 settings.database.url（POSTGRES_DB_URL / DATABASE_URL）。"""
        from multi_agent_research.core.config import settings
        return settings.database.url

    @classmethod
    def _redis_url(cls) -> str:
        """Read the Redis connection URL from the canonical Redis settings."""
        from multi_agent_research.core.config import settings
        return settings.redis.url

    @classmethod
    def _compute_cache_key(cls, backend: str) -> str:
        """
        在锁外预计算含连接参数的精确 key。

        保证：同一 backend 不同连接目标不复用同一实例；
        进入锁后只需 dict.get(key) 一次，无需迭代。
        """
        if backend == "sqlite":
            return f"sqlite:{cls._sqlite_db_path()}"
        if backend == "postgres":
            return f"postgres:{cls._postgres_conn_str()}"
        if backend == "redis":
            return f"redis:{cls._redis_url()}"
        return backend  # "memory"

    @staticmethod
    def _safe_key_label(key: str) -> str:
        """Return a log-safe lifecycle label without connection credentials."""
        return key.split(":", 1)[0]

    @classmethod
    def _register_lifecycle(
        cls, key: str, *, cm: Any = None, pool: Any = None
    ) -> None:
        """注册需要在 close_all() 时清理的外部资源（必须在锁内调用）。"""
        cls._lifecycle[key] = {"cm": cm, "pool": pool}

    # ── 生命周期管理 ──────────────────────────────────────────────────────────

    @classmethod
    async def close_all(cls) -> None:
        """
        关闭所有已注册连接资源，完全重置类状态。

        供 FastAPI lifespan 的 shutdown 阶段调用：
            @asynccontextmanager
            async def lifespan(app):
                yield
                await CheckpointerFactory.close_all()
        """
        lock = cls._get_lock()
        async with lock:
            for key, resources in cls._lifecycle.items():
                safe_key = cls._safe_key_label(key)
                cm = resources.get("cm")
                if cm is not None:
                    try:
                        await cm.__aexit__(None, None, None)
                        cls._log(f"[Checkpointer] 已关闭 CM: {safe_key}")
                    except Exception as e:
                        cls._log(
                            f"[Checkpointer] 关闭 CM 失败 ({safe_key}): {type(e).__name__}",
                            "warning",
                        )

                pool = resources.get("pool")
                if pool is not None:
                    try:
                        await pool.close()
                        cls._log(f"[Checkpointer] 已关闭连接池: {safe_key}")
                    except Exception as e:
                        cls._log(
                            f"[Checkpointer] 关闭连接池失败 ({safe_key}): {type(e).__name__}",
                            "warning",
                        )

            cls._instances.clear()
            cls._effective.clear()
            cls._lifecycle.clear()

        # 退出 async with 后再重置锁，下次在新 loop 中重建
        cls._lock = None

    # ── 对外入口 ──────────────────────────────────────────────────────────────

    @classmethod
    async def create(cls, *, require_durable: bool = False) -> Optional[BaseCheckpointSaver]:
        """
        获取（或创建）Checkpointer 实例。

        快速路径（无锁）：CPython dict.get 是原子操作，已初始化时直接返回。
        慢路径（加锁）  ：双重检查，确保只创建一次。
        """
        from multi_agent_research.core.config import settings
        backend = cls._normalize_backend(settings.agent.checkpoint_backend)

        if require_durable and backend not in {"sqlite", "postgres"}:
            raise RuntimeError("API 要求可恢复的 SQLite/PostgreSQL Checkpoint；不允许 memory/none 或未知后端")

        if backend == "none":
            return None

        # 在锁外预计算 key（涉及路径解析 / settings 读取，成本低但避免在锁内做）
        cache_key = cls._compute_cache_key(backend)

        # 快速路径
        inst = cls._instances.get(cache_key)
        if inst is not None:
            cls._check_durable(cache_key, require_durable)
            return inst

        # 慢路径
        async with cls._get_lock():
            # 双重检查（等待锁期间可能已被其他协程创建）
            inst = cls._instances.get(cache_key)
            if inst is not None:
                cls._check_durable(cache_key, require_durable)
                return inst

            instance, effective_key = await cls._create_instance(backend, cache_key)
            cls._effective[cache_key] = effective_key.split(":", 1)[0] if effective_key else "none"

            if instance is not None and effective_key:
                cls._instances[effective_key] = instance
                # 降级时同时在原始 key 下缓存，防止后续请求重复尝试已知失败的 backend
                if effective_key != cache_key:
                    cls._instances[cache_key] = instance

            cls._check_durable(cache_key, require_durable)
            return instance

    @classmethod
    def _check_durable(cls, key, required):
        if required and cls._effective.get(key) not in {"sqlite", "postgres"}:
            raise RuntimeError("Checkpoint 持久化初始化失败；API 拒绝降级内存，请修复存储配置后重启")

    # ── 各 Backend 创建逻辑 ───────────────────────────────────────────────────

    @classmethod
    async def _create_instance(
        cls, backend: str, cache_key: str
    ) -> Tuple[Optional[BaseCheckpointSaver], str]:
        """分发到具体 backend 创建函数（在锁内调用）。"""
        if backend == "memory":
            return cls._create_memory(), "memory"
        if backend == "sqlite":
            return await cls._create_sqlite(cache_key)
        if backend == "postgres":
            return await cls._create_postgres(cache_key)
        if backend == "redis":
            return await cls._create_redis(cache_key)
        cls._log(f"[Checkpointer] 未知 backend '{backend}'，禁用持久化", "warning")
        return None, ""

    @classmethod
    def _create_memory(cls) -> Optional[BaseCheckpointSaver]:
        try:
            from langgraph.checkpoint.memory import MemorySaver
            cls._log(
                "[Checkpointer] 使用 MemorySaver（仅限本地调试，"
                "无持久化，高并发存在 OOM 风险，禁止用于生产环境）",
                "warning",
            )
            inst = MemorySaver()
            cls._register_lifecycle("memory")  # 无外部资源，注册空记录保持注册表完整
            return inst
        except ImportError as e:
            cls._log(f"[Checkpointer] ❌ 无法导入 MemorySaver: {e}", "error")
            return None

    @classmethod
    def _fallback_to_memory(
        cls, reason: str
    ) -> Tuple[Optional[BaseCheckpointSaver], str]:
        """
        统一降级入口（必须在锁内调用）。

        所有 backend 失败共享同一个 MemorySaver 实例（"memory" key），
        避免多次降级各自创建独立实例导致同一 thread_id 看到不同会话历史。
        """
        cls._log(f"[Checkpointer] ⚠️ {reason}，降级到 MemorySaver", "warning")
        existing = cls._instances.get("memory")
        if existing is not None:
            return existing, "memory"
        inst = cls._create_memory()
        return (inst, "memory") if inst is not None else (None, "")

    @classmethod
    async def _create_sqlite(
        cls, ok_key: str
    ) -> Tuple[Optional[BaseCheckpointSaver], str]:
        db_path = cls._sqlite_db_path()
        try:
            from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
        except ImportError as e:
            return cls._fallback_to_memory(f"AsyncSqliteSaver 不可用: {e}")

        db_path.parent.mkdir(parents=True, exist_ok=True)

        cm = None
        entered = False
        try:
            cls._log(f"[Checkpointer] 初始化 SQLite: {db_path}")
            cm = AsyncSqliteSaver.from_conn_string(str(db_path))
            checkpointer = await cm.__aenter__()
            entered = True
            await asyncio.wait_for(checkpointer.setup(), timeout=20)
            cls._register_lifecycle(ok_key, cm=cm)
            cls._log(f"[Checkpointer] ✅ SQLite 已启用: {db_path}")
            return checkpointer, ok_key
        except BaseException as e:
            if cm is not None and entered:
                await cm.__aexit__(None, None, None)
            if not isinstance(e, Exception):
                raise
            cls._log(
                f"[Checkpointer] ❌ SQLite 初始化失败: {type(e).__name__}: {e}", "error"
            )
            return cls._fallback_to_memory("SQLite 初始化失败")

    @classmethod
    async def _create_postgres(
        cls, ok_key: str
    ) -> Tuple[Optional[BaseCheckpointSaver], str]:
        if AsyncPostgresSaver is None or AsyncConnectionPool is None:
            return cls._fallback_to_memory(
                "缺少依赖：pip install langgraph-checkpoint-postgres psycopg[binary,pool]"
            )

        conn_str = cls._postgres_conn_str()

        from multi_agent_research.core.config import settings
        db_cfg = settings.database
        max_size: int         = db_cfg.pool_size if db_cfg else 5
        connect_timeout: float = 5.0

        pool = None
        try:
            cls._log("[Checkpointer] 初始化 Postgres 连接池...")
            pool = AsyncConnectionPool(
                conn_str,
                max_size=max_size,
                kwargs={"autocommit": True},
                open=False,
            )
            # wait_for 包裹 pool.open()：Postgres 不可达时防止永久挂起
            await asyncio.wait_for(pool.open(wait=True), timeout=connect_timeout)
            checkpointer = AsyncPostgresSaver(pool)
            await asyncio.wait_for(checkpointer.setup(), timeout=20)
            cls._register_lifecycle(ok_key, pool=pool)
            cls._log("[Checkpointer] ✅ Postgres 已启用")
            return checkpointer, ok_key

        except asyncio.TimeoutError:
            if pool is not None:
                try:
                    await pool.close()
                except Exception:
                    pass
            return cls._fallback_to_memory(
                f"Postgres 连接池超时（{connect_timeout}s），请检查 POSTGRES_DB_URL 和网络连通性"
            )
        except asyncio.CancelledError:
            if pool is not None:
                await pool.close()
            raise
        except Exception as e:
            if pool is not None:
                try:
                    await pool.close()
                except Exception:
                    pass
            cls._log(
                f"[Checkpointer] ❌ Postgres 初始化失败: {type(e).__name__}: {e}", "error"
            )
            return cls._fallback_to_memory("Postgres 初始化失败")

    @classmethod
    async def _create_redis(
        cls, ok_key: str
    ) -> Tuple[Optional[BaseCheckpointSaver], str]:
        try:
            from langgraph.checkpoint.redis.aio import AsyncRedisSaver
        except ImportError:
            return cls._fallback_to_memory(
                "缺少依赖：pip install langgraph-checkpoint-redis"
            )

        redis_url = cls._redis_url()
        try:
            cls._log(f"[Checkpointer] 初始化 Redis: {redis_url}")
            cm = AsyncRedisSaver.from_conn_string(redis_url)
            checkpointer = await cm.__aenter__()
            await checkpointer.setup()
            cls._register_lifecycle(ok_key, cm=cm)
            cls._log("[Checkpointer] ✅ Redis 已启用")
            return checkpointer, ok_key
        except Exception as e:
            cls._log(
                f"[Checkpointer] ❌ Redis 初始化失败: {type(e).__name__}: {e}", "error"
            )
            return cls._fallback_to_memory("Redis 初始化失败")

    # ── 健康检查（供 /api/health 端点消费）───────────────────────────────────

    @classmethod
    async def health_check(cls) -> Dict[str, Any]:
        """
        探活所有已注册 backend，返回脱敏的后端状态字典。

        使用不存在的 thread_id 做轻量探针，不产生副作用。
        """
        if not cls._instances:
            return {"status": "uninitialized", "persistent": False, "backend": "none"}

        result: Dict[str, str] = {}
        probe_config = {"configurable": {"thread_id": "__health_probe__"}}
        for key, inst in cls._instances.items():
            backend = key.split(":", 1)[0]
            try:
                await inst.aget_tuple(probe_config)
                result[backend] = "ok"
            except Exception as e:
                result[backend] = f"error: {type(e).__name__}"
        effective = set(cls._effective.values()) or {"memory"}
        persistent = bool(effective) and effective <= {"sqlite", "postgres"}
        return {**result, "status": "ok" if all(value == "ok" for value in result.values()) else "error",
                "persistent": persistent, "backend": ",".join(sorted(effective))}
