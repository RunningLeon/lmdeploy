# Copyright (c) OpenMMLab. All rights reserved.
"""Normal shutdown is ordered work; a failed rank retains forced termination."""
import asyncio
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from lmdeploy.pytorch.engine.executor.mp_executor import ExecutorProc, MPExecutor


@pytest.mark.parametrize('graceful,alive', [(True, True), (True, False), (False, True)])
def test_mp_release_orders_shutdown_before_join_and_buffer_close(graceful, alive):
    events = []
    executor = MPExecutor.__new__(MPExecutor)
    executor.procs = [SimpleNamespace(
        _proc=SimpleNamespace(is_alive=lambda: alive),
        close=lambda: events.append('terminate'),
        join=lambda: events.append('join')) for _ in range(2)]
    executor.comm_buf = SimpleNamespace(close=lambda: events.append('close_comm'))
    executor.ret_bufs = [SimpleNamespace(close=lambda: events.append('close_return'))]
    executor.collective_rpc = Mock(side_effect=lambda *a, **kw: events.append('shutdown'))
    executor.release(graceful=graceful)
    if graceful and alive:
        executor.collective_rpc.assert_called_once_with('_shutdown', return_mask=0)
        assert events == ['shutdown', 'join', 'join', 'close_comm', 'close_return']
    else:
        executor.collective_rpc.assert_not_called()
        assert events == ['terminate', 'terminate', 'join', 'join', 'close_comm', 'close_return']


def test_mp_shutdown_awaits_worker_stop_without_return_buffer_reply():
    async def run():
        events = []

        async def receive():
            events.append('receive')
            return {'method': '_shutdown', 'return_mask': 0}

        async def stop():
            events.append('stop_begin')
            await asyncio.sleep(0)
            events.append('stop_done')

        worker = SimpleNamespace(stop_async=stop)
        ret_buf = Mock()
        await ExecutorProc._main_loop_impl(
            None, 0, SimpleNamespace(receive_async=receive), ret_buf, worker)
        assert events == ['receive', 'stop_begin', 'stop_done']
        assert not ret_buf.mock_calls
    asyncio.run(run())
