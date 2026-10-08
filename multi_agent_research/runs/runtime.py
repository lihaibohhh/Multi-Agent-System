"""Database-scoped single-instance ownership, held on a dedicated connection."""

import asyncio

from psycopg import AsyncConnection
from psycopg.rows import dict_row


class InstanceUnavailableError(RuntimeError):
    pass


class InstanceLock:
    def __init__(self, key: tuple[int, int] = (1296126534, 546132)):
        self.key = key
        self.connection = None
        self._mutex = asyncio.Lock()

    async def acquire(self, conninfo: str):
        async with self._mutex:
            if self.connection is not None:
                raise InstanceUnavailableError("实例所有权已经初始化")
            conn = await AsyncConnection.connect(conninfo, autocommit=True, row_factory=dict_row,
                connect_timeout=5, options="-c statement_timeout=5000")
            try:
                row = await (await conn.execute("SELECT pg_try_advisory_lock(%s, %s) AS acquired", self.key)).fetchone()
                if not row["acquired"]:
                    raise InstanceUnavailableError("同一数据库已有研究 API 实例运行；仅支持单实例、单 worker")
                self.connection = conn
            except BaseException:
                await conn.close()
                raise

    async def check(self):
        async with self._mutex:
            if self.connection is None or self.connection.closed:
                raise InstanceUnavailableError("实例数据库锁不可用；需重启服务")
            try:
                async with asyncio.timeout(6):
                    row = await (await self.connection.execute("""
                        SELECT EXISTS(SELECT 1 FROM pg_locks WHERE locktype = 'advisory'
                            AND pid = pg_backend_pid() AND classid = %s AND objid = %s
                            AND objsubid = 2 AND granted) AS owned
                    """, self.key)).fetchone()
                if not row["owned"]:
                    raise InstanceUnavailableError("实例数据库锁已失去；需重启服务")
            except Exception as exc:
                raise InstanceUnavailableError("无法确认实例所有权；已停止接受执行请求，请检查数据库并重启服务") from exc

    async def release(self):
        async with self._mutex:
            if self.connection is not None:
                await self.connection.close()
                self.connection = None
