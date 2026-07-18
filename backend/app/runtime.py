from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from typing import Any, Callable, TypeVar

from app.config import Settings


Result = TypeVar("Result")


class CapacityGate:
    """A process-local active-work limit with a small, explicit waiting room."""

    def __init__(self, capacity: int, max_waiters: int, timeout_seconds: float):
        self._semaphore = asyncio.Semaphore(capacity)
        self.capacity = capacity
        self.max_waiters = max_waiters
        self.timeout_seconds = timeout_seconds
        self.active = 0
        self.waiting = 0

    async def acquire(self) -> bool:
        if self._semaphore.locked() and self.waiting >= self.max_waiters:
            return False

        self.waiting += 1
        try:
            await asyncio.wait_for(
                self._semaphore.acquire(), timeout=self.timeout_seconds
            )
        except TimeoutError:
            return False
        finally:
            self.waiting -= 1

        self.active += 1
        return True

    def release(self) -> None:
        if self.active <= 0:
            raise RuntimeError("capacity released without a matching acquisition")
        self.active -= 1
        self._semaphore.release()


class ComputeRuntime:
    def __init__(self, config: Settings):
        self.config = config
        self.gate = CapacityGate(
            config.max_compute_jobs,
            config.max_compute_waiters,
            config.admission_timeout_seconds,
        )
        self.executor = ThreadPoolExecutor(
            max_workers=config.compute_workers,
            thread_name_prefix="turing-compute",
        )

    async def run(self, function: Callable[..., Result], *args: Any) -> Result:
        """Run bounded CPU work without abandoning it when its await is cancelled."""

        loop = asyncio.get_running_loop()
        future = loop.run_in_executor(self.executor, partial(function, *args))
        try:
            return await asyncio.shield(future)
        except asyncio.CancelledError:
            # Native NumPy/Pillow work cannot be cancelled safely. Waiting here keeps
            # the owning capacity slot held until the real work has stopped.
            try:
                await future
            finally:
                raise

    def close(self) -> None:
        self.executor.shutdown(wait=True, cancel_futures=True)
