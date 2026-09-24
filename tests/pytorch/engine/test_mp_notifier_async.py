# Copyright (c) OpenMMLab. All rights reserved.
"""MP ring-wrap barriers must not starve pending model-forward tasks."""
import asyncio
import threading

import pytest

from lmdeploy.pytorch.engine.executor.mp_executor import NUM_SHARED_BLOCK, Notifier


@pytest.mark.parametrize('operation', ['receive', 'send'])
@pytest.mark.parametrize('delayed_phase', [1, 2])
def test_wrap_barriers_allow_local_async_forward_progress(operation, delayed_phase):
    forward_ready = threading.Event()

    class PeerBarrier:
        calls = 0

        def wait(self):
            self.calls += 1
            if self.calls == delayed_phase:
                # A peer cannot reach this barrier until the local model task
                # issues the collective it needs. A blocking event-loop wait
                # deadlocks this dependency; timeout makes the test fail safely.
                assert forward_ready.wait(1), 'ring barrier starved the local forward task'

    notifier = Notifier.__new__(Notifier)
    notifier.events = [threading.Event() for _ in range(NUM_SHARED_BLOCK)]
    notifier.events[-1].set()
    notifier._event_id = NUM_SHARED_BLOCK - 1
    notifier.bar = PeerBarrier()

    async def run():
        async def forward():
            await asyncio.sleep(.05)
            forward_ready.set()
        task = asyncio.create_task(forward())
        try:
            if operation == 'receive':
                async with notifier.wait_async():
                    pass
            else:
                await notifier.set_async()
            await task
        finally:
            if not task.done():
                task.cancel()

    asyncio.run(run())
    assert notifier.bar.calls == 2
    assert notifier._event_id == 0
    if operation == 'send':
        assert not any(event.is_set() for event in notifier.events)


@pytest.mark.parametrize('operation', ['receive', 'send'])
def test_nonwrap_does_not_enter_barrier(operation):
    notifier = Notifier.__new__(Notifier)
    notifier.events = [threading.Event() for _ in range(NUM_SHARED_BLOCK)]
    notifier.events[0].set()
    notifier._event_id = 0
    notifier.bar = None

    async def run():
        if operation == 'receive':
            async with notifier.wait_async():
                pass
        else:
            await notifier.set_async()

    asyncio.run(run())
    assert notifier._event_id == 1
