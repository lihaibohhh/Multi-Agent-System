"""Fence asynchronous checkpoint writes by the business execution generation."""

import asyncio
from contextvars import ContextVar

from langgraph.checkpoint.base import BaseCheckpointSaver


# RunService binds (repository, run_id, execution_id); graph background writes
# inherit it. Standalone CLI graphs have no business Run to fence.
execution_fence: ContextVar = ContextVar("execution_fence", default=None)


class FencedCheckpointer(BaseCheckpointSaver):
    def __init__(self, delegate, *, write_timeout: float = 30):
        super().__init__(serde=delegate.serde)
        self.delegate = delegate
        self.write_timeout = write_timeout

    @property
    def config_specs(self):
        return self.delegate.config_specs

    def get_next_version(self, current, channel):
        return self.delegate.get_next_version(current, channel)

    async def aget_tuple(self, config):
        return await self.delegate.aget_tuple(config)

    async def alist(self, config, *, filter=None, before=None, limit=None):
        async for item in self.delegate.alist(config, filter=filter, before=before, limit=limit):
            yield item

    async def _write(self, config, operation):
        fence = execution_fence.get()
        if fence is None:
            return await operation()
        repository, run_id, execution_id = fence
        if config["configurable"]["thread_id"] != run_id:
            raise RuntimeError("checkpoint thread does not match execution owner")
        # Keep the business row locked until the checkpoint write completes.
        # A replacement execution cannot be claimed between the check and write.
        async with asyncio.timeout(self.write_timeout):
            async with repository.guard_execution(run_id, execution_id):
                return await operation()

    async def aput(self, config, checkpoint, metadata, new_versions):
        return await self._write(config, lambda: self.delegate.aput(config, checkpoint, metadata, new_versions))

    async def aput_writes(self, config, writes, task_id, task_path=""):
        return await self._write(config, lambda: self.delegate.aput_writes(config, writes, task_id, task_path))
