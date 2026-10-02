# multi_agent_research/utils/redis_client.py
from __future__ import annotations
import threading
import redis.asyncio as aioredis
import redis as sync_redis
import logging
from ..core.config import settings


logger = logging.getLogger(__name__)
_async_pool: aioredis.ConnectionPool | None = None
_sync_pool: sync_redis.ConnectionPool | None = None
_lock = threading.Lock()


def get_async_redis() -> aioredis.Redis:
    """获取异步 Redis 客户端（用于 async 代码）"""
    global _async_pool
    if _async_pool is None:
        with _lock:
            if _async_pool is None:
                _async_pool = aioredis.ConnectionPool.from_url(
                    settings.redis.url,
                    max_connections=settings.redis.max_connections,
                    decode_responses=False,   # 存 pickle 时用 bytes，不要 decode
                )
    return aioredis.Redis(connection_pool=_async_pool)


def get_sync_redis() -> sync_redis.Redis:
    """获取同步 Redis 客户端（用于初始化阶段的阻塞代码）"""
    global _sync_pool
    if _sync_pool is None:
        with _lock:
            if _sync_pool is None:
                _sync_pool = sync_redis.ConnectionPool.from_url(
                    settings.redis.url,
                    max_connections=settings.redis.max_connections,
                    decode_responses=False,
                )
    return sync_redis.Redis(connection_pool=_sync_pool)


async def ping_redis() -> bool:
    """启动时健康检查，失败时 fallback 逻辑用这个"""
    try:
        r = get_async_redis()
        await r.ping()
        return True
    except Exception as e:
        logger.info(f"[Redis] ⚠️ 连接失败: {e}，将降级为本地模式")
        return False
