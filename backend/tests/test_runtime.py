import asyncio

import pytest

from app.runtime import CapacityGate


def test_capacity_gate_rejects_excess_work_and_releases_slots():
    async def scenario():
        gate = CapacityGate(capacity=1, max_waiters=0, timeout_seconds=0.01)

        assert await gate.acquire() is True
        assert gate.active == 1
        assert await gate.acquire() is False
        assert gate.waiting == 0

        gate.release()
        assert gate.active == 0
        assert await gate.acquire() is True
        gate.release()

    asyncio.run(scenario())


def test_capacity_gate_refuses_unmatched_release():
    gate = CapacityGate(capacity=1, max_waiters=0, timeout_seconds=0.01)

    with pytest.raises(RuntimeError):
        gate.release()
