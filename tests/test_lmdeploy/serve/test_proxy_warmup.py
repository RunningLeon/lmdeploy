# Copyright (c) OpenMMLab. All rights reserved.
import asyncio
from collections import Counter
from types import SimpleNamespace

import httpx
from aiohttp import web

from lmdeploy.pytorch.disagg.config import DistServeEngineConfig
from lmdeploy.pytorch.disagg.conn.proxy_conn import PDConnectionPool
from lmdeploy.serve.proxy import proxy


def test_warmup_http_bounded_timeout_and_recovery(monkeypatch):
    """Real aiohttp handshakes, ASGI proxy entry, 10 x 10 nodes and 40
    callers."""
    async def run():
        counts = Counter()
        first_request = asyncio.Event()
        release = asyncio.Event()
        config = DistServeEngineConfig(tp_size=1, ep_size=1, dp_size=1, pp_size=1, dp_rank=0,
                                       block_size=16, num_cpu_blocks=0, num_gpu_blocks=8)

        async def backend(request):
            counts[request.match_info['api']] += 1
            first_request.set()
            await release.wait()
            api = request.match_info['api']
            if api == 'engine_info':
                return web.json_response(config.model_dump_json())
            if api == 'p2p_initialize':
                return web.json_response({'status': 1, 'engine_endpoint_info': {'zmq_address': 'unused'},
                                          'kvtransfer_endpoint_info': []})
            return web.json_response({'status': 1})

        app = web.Application()
        app.router.add_route('*', '/{node}/distserve/{api}', backend)
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, '127.0.0.1', 0)
        await site.start()
        port = site._server.sockets[0].getsockname()[1]
        base = f'http://127.0.0.1:{port}'
        pool = PDConnectionPool()
        manager = SimpleNamespace(
            pd_connection_pool=pool, connection_warmup_lock=asyncio.Lock(), connection_warmup_timeout=2,
            prefill_nodes={f'{base}/p{i}': None for i in range(10)},
            decode_nodes={f'{base}/d{i}': None for i in range(10)},
            migration_protocol=proxy.MigrationProtocol.RDMA, rdma_config=None)
        monkeypatch.setattr(proxy, 'node_manager', manager)
        try:
            transport = httpx.ASGITransport(app=proxy.app)
            async with httpx.AsyncClient(transport=transport, base_url='http://proxy') as client:
                first = asyncio.create_task(client.post('/distserve/connection_warmup', json={}))
                await asyncio.wait_for(first_request.wait(), 2)
                others = await asyncio.gather(
                    *[client.post('/distserve/connection_warmup', json={}) for _ in range(39)])
                assert all(response.status_code == 409 for response in others)
                assert len(pool.pool) == pool._connect_requests == 32
                response = await asyncio.wait_for(first, 4)
                assert response.status_code == 504
                assert not pool.pool and pool._connect_requests == 0
                assert not manager.connection_warmup_lock.locked()
                release.set()
                manager.connection_warmup_timeout = 10
                response = await client.post('/distserve/connection_warmup', json={})
                assert response.status_code == 200 and response.json() == {'SUCCESS': True}
                assert len(pool.pool) == 100
                assert all(pool.is_connected(p, d) for p in manager.prefill_nodes for d in manager.decode_nodes)
                assert counts['p2p_initialize'] == counts['p2p_connect'] == 200
                before = counts.copy()
                assert (await client.post('/distserve/connection_warmup', json={})).status_code == 200
                assert counts == before
        finally:
            await pool.close()
            release.set()
            await runner.cleanup()

    asyncio.run(run())


def test_warmup_failure_cancels_siblings_and_releases_round(monkeypatch):
    async def run():
        pool = PDConnectionPool()
        pool.max_retry_cnt = 1
        sibling_started, sibling_stopped = asyncio.Event(), asyncio.Event()

        async def handshake(req):
            if req.d_url == 'fail':
                await sibling_started.wait()
                raise RuntimeError('backend initialization failed')
            sibling_started.set()
            try:
                await asyncio.Event().wait()
            finally:
                sibling_stopped.set()

        monkeypatch.setattr(pool, '_connect_once', handshake)
        manager = SimpleNamespace(
            pd_connection_pool=pool, connection_warmup_lock=asyncio.Lock(), connection_warmup_timeout=2,
            prefill_nodes={'p': None}, decode_nodes={'fail': None, 'blocked': None},
            migration_protocol=proxy.MigrationProtocol.RDMA, rdma_config=None)
        monkeypatch.setattr(proxy, 'node_manager', manager)
        try:
            response = await asyncio.wait_for(proxy.connection_warmup(), 3)
            assert response.status_code == 503
            assert sibling_stopped.is_set()
            assert not pool.pool and pool._connect_requests == 0
            assert not manager.connection_warmup_lock.locked()
            # An empty topology is a successful no-op, with no allocated work.
            manager.decode_nodes = {}
            assert (await proxy.connection_warmup()).status_code == 200
        finally:
            await pool.close()

    asyncio.run(run())


def test_warmup_cancellation_and_proxy_lifespan_cleanup(monkeypatch):
    async def run():
        pool = PDConnectionPool()
        started = asyncio.Event()

        async def handshake(req):
            started.set()
            await asyncio.Event().wait()

        monkeypatch.setattr(pool, '_connect_once', handshake)
        manager = SimpleNamespace(
            pd_connection_pool=pool, connection_warmup_lock=asyncio.Lock(), connection_warmup_timeout=2,
            prefill_nodes={'p': None}, decode_nodes={'d': None},
            migration_protocol=proxy.MigrationProtocol.RDMA, rdma_config=None)
        monkeypatch.setattr(proxy, 'node_manager', manager)
        async with proxy.lifespan(proxy.app):
            caller = asyncio.create_task(proxy.connection_warmup())
            await asyncio.wait_for(started.wait(), 1)
            caller.cancel()
            result, = await asyncio.wait_for(asyncio.gather(caller, return_exceptions=True), 1)
            assert isinstance(result, asyncio.CancelledError)
            assert not pool.pool and pool._connect_requests == 0
            assert not manager.connection_warmup_lock.locked()
            assert not pool.conn_sess.closed
        assert pool.conn_sess.closed
        assert pool._closed

    asyncio.run(run())
