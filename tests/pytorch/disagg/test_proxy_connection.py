# Copyright (c) OpenMMLab. All rights reserved.
import asyncio

import pytest

from lmdeploy.pytorch.disagg.conn.proxy_conn import PDConnectionPool
from lmdeploy.pytorch.disagg.messages import PDConnectionMessage


def test_connection_single_flight_admission_and_cancellation(monkeypatch):
    async def run():
        pool = PDConnectionPool()
        pool.max_connect_requests = 8
        started, finish = asyncio.Event(), asyncio.Event()
        calls = 0

        async def handshake(req):
            nonlocal calls
            calls += 1
            started.set()
            await finish.wait()

        monkeypatch.setattr(pool, '_connect_once', handshake)
        req = PDConnectionMessage(p_url='p', d_url='d')
        callers = [asyncio.create_task(pool.connect(req)) for _ in range(40)]
        try:
            await asyncio.wait_for(started.wait(), 1)
            assert pool._connect_requests == 8
            assert len(pool.pool) == calls == 1
            rejected = await asyncio.gather(*callers[8:], return_exceptions=True)
            assert all(isinstance(exc, RuntimeError) for exc in rejected)
            callers[0].cancel()
            await asyncio.gather(callers[0], return_exceptions=True)
            assert not pool.pool['p', 'd'].task.cancelled()
            finish.set()
            await asyncio.gather(*callers[1:8])
            assert pool.is_connected('p', 'd')
            await pool.connect(req)
            assert calls == 1
            assert pool._connect_requests == 0
        finally:
            await pool.close()
            await asyncio.gather(*callers, return_exceptions=True)
        assert pool.conn_sess.closed

    asyncio.run(run())


@pytest.mark.parametrize('failure', ['timeout', 'error', 'cancel'])
def test_connection_failure_cleanup_and_retry(monkeypatch, failure):
    async def run():
        pool = PDConnectionPool()
        pool.connection_timeout = 0.05
        pool.max_retry_cnt = 2
        started = asyncio.Event()
        attempts = 0

        async def handshake(req):
            nonlocal attempts
            attempts += 1
            started.set()
            if failure == 'error':
                raise ValueError('backend failure')
            await asyncio.Event().wait()

        monkeypatch.setattr(pool, '_connect_once', handshake)
        req = PDConnectionMessage(p_url='p', d_url='d')
        try:
            caller = asyncio.create_task(pool.connect(req))
            await asyncio.wait_for(started.wait(), 1)
            if failure == 'cancel':
                caller.cancel()
            result, = await asyncio.wait_for(asyncio.gather(caller, return_exceptions=True), 1)
            expected = {'timeout': asyncio.TimeoutError, 'error': ConnectionError, 'cancel': asyncio.CancelledError}
            assert isinstance(result, expected[failure])
            assert not pool.pool
            assert pool._connect_requests == 0
            assert attempts == (2 if failure == 'error' else 1)

            async def success(req):
                pass

            monkeypatch.setattr(pool, '_connect_once', success)
            await pool.connect(req)
            assert pool.is_connected('p', 'd')
        finally:
            await pool.close()

    asyncio.run(run())


def test_drop_and_shutdown_cancel_live_handshakes(monkeypatch):
    async def run():
        pool = PDConnectionPool()
        started = asyncio.Event()

        async def handshake(req):
            started.set()
            await asyncio.Event().wait()

        monkeypatch.setattr(pool, '_connect_once', handshake)
        monkeypatch.setattr('lmdeploy.pytorch.disagg.conn.proxy_conn.requests.post', lambda *a, **k: None)
        caller = asyncio.create_task(pool.connect(PDConnectionMessage(p_url='p', d_url='d')))
        await asyncio.wait_for(started.wait(), 1)
        pool.drop(('p', 'd'))
        await asyncio.wait_for(asyncio.gather(caller, return_exceptions=True), 1)
        assert not pool.pool and pool._connect_requests == 0
        started.clear()
        caller = asyncio.create_task(pool.connect(PDConnectionMessage(p_url='p', d_url='d')))
        await asyncio.wait_for(started.wait(), 1)
        await pool.close()
        await asyncio.wait_for(asyncio.gather(caller, return_exceptions=True), 1)
        assert not pool.pool and pool._connect_requests == 0 and pool.conn_sess.closed

    asyncio.run(run())


@pytest.mark.parametrize('name, value', [('LMDEPLOY_PD_CONNECTION_TIMEOUT', '0'),
                                        ('LMDEPLOY_PD_CONNECTION_TIMEOUT', 'inf'),
                                        ('LMDEPLOY_PD_CONNECTION_TIMEOUT', 'nan'),
                                        ('LMDEPLOY_PD_MAX_CONNECT_REQUESTS', '0')])
def test_connection_limits_require_finite_positive_values(monkeypatch, name, value):
    monkeypatch.setenv(name, value)
    with pytest.raises(ValueError):
        PDConnectionPool()
