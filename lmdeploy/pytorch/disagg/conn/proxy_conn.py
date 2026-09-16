# Copyright (c) OpenMMLab. All rights reserved.
import asyncio
import enum
import math
import os
from collections import defaultdict

import aiohttp
import requests

from lmdeploy.logger import get_logger
from lmdeploy.pytorch.disagg.config import DistServeEngineConfig, EngineRole
from lmdeploy.pytorch.disagg.conn.protocol import (
    DistServeCacheFreeRequest,
    DistServeConnectionRequest,
    DistServeConnectionResponse,
    DistServeConnectionStatus,
    DistServeDropConnectionRequest,
    DistServeInitRequest,
    DistServeInitResponse,
)
from lmdeploy.pytorch.disagg.messages import PDConnectionMessage

logger = get_logger('lmdeploy')


def positive_env(name: str, default: float) -> float:
    """Read a finite, positive resource deadline from the environment."""
    value = float(os.getenv(name, default))
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f'{name} must be finite and positive')
    return value


class PDConnectionStatus(enum.Enum):
    Disconnected = enum.auto()
    Connected = enum.auto()
    Connecting = enum.auto()


class PDConnectionState:
    """PDConnectionState."""

    def __init__(self):
        self.status = PDConnectionStatus.Connecting
        self.task: asyncio.Task | None = None
        self.waiters = 0


def get_server_api(url: str, api: str):
    return f'{url}/{api}'


class PDConnectionPool:
    """Constructing the link of Prefill and Decode engine for the migration of
    KVCache.

    Note: we use Peer to Peer transportation in KVCache migration.
    Note: Lazy link construction is supported, which perform connection
        at the first LLM request. As a result, we don't need to construct
        PD Communication group when start a engine server.
    Note: we perform simple fault tolerance by checkpointing the session_id of a
        request which is under migrating and will trigger `gc` when the decode
        instanceis crushed.
    TODO (JimyMa): By now, only engines with same parallel configuration can be
        correctly connected.
    """

    # Maximum concurrent connections​​
    CONN_SEMAPHORE_SIZE = 2048

    def __init__(self):
        # all prefill and decode instances
        # TODO (JimyMa): Maybe encoding instances
        self.prefill_endpoints: set[str] = set()
        self.decode_endpoints: set[str] = set()

        # Links of PD Connection.
        self.pool: dict[tuple[str, str], PDConnectionState] = {}

        # put migrating session to `self.migration_session_shelf` for increasing fault tolerance
        # if a session is finished, then pop it from `self.migration_session_shelf`
        # if a decode instance is disconnected, then gc all blocks of these sessions in prefill instance.
        self.migration_session_shelf: dict[tuple[str, str], set[int]] = defaultdict(set)

        # Admission bounds callers, not just sockets or active handshakes.
        self.max_connect_requests = int(os.getenv('LMDEPLOY_PD_MAX_CONNECT_REQUESTS', 2048))
        if self.max_connect_requests < 1:
            raise ValueError('LMDEPLOY_PD_MAX_CONNECT_REQUESTS must be at least 1')
        self.connection_timeout = positive_env('LMDEPLOY_PD_CONNECTION_TIMEOUT', 60)
        self.max_retry_cnt = 8
        self._connect_requests = 0
        self.conn_sess: aiohttp.ClientSession | None = None
        self._closed = False

    def reg_instance(self, role: EngineRole, endpoint: str):
        if role == EngineRole.Prefill:
            self.prefill_endpoints.add(endpoint)
        elif role == EngineRole.Decode:
            self.decode_endpoints.add(endpoint)
        else:
            raise ValueError(f'Unsupported role: {role}')

    def dereg_instance(self, endpoint: str):
        for key in list(self.pool):
            if endpoint in key:
                self.drop(key)
        self.prefill_endpoints.discard(endpoint)
        self.decode_endpoints.discard(endpoint)

    def shelf_prefill_session(self, conn_key: tuple[str, str], session_id: int):
        self.migration_session_shelf[conn_key].add(session_id)

    def unshelf_prefill_session(self, conn_key: tuple[str, str], session_id: int):
        self.migration_session_shelf[conn_key].remove(session_id)

    async def connect(self, conn_req: PDConnectionMessage):
        """Share one bounded-lifetime handshake per pair, with no waiter
        tasks."""
        if self._closed:
            raise RuntimeError('PD connection pool is closed')
        link = (conn_req.p_url, conn_req.d_url)
        if self.is_connected(*link):
            return
        if self._connect_requests >= self.max_connect_requests:
            raise RuntimeError('PD connection pool is busy; retry later')
        if self.conn_sess is None:
            self.conn_sem = asyncio.Semaphore(self.CONN_SEMAPHORE_SIZE)
            self.aiotimeout = aiohttp.ClientTimeout(total=self.connection_timeout)
            self.conn_sess = aiohttp.ClientSession(
                connector=aiohttp.TCPConnector(limit_per_host=256), timeout=self.aiotimeout)

        state = self.pool.get(link)
        if state is not None and state.status == PDConnectionStatus.Disconnected:
            raise RuntimeError('PD connection is being cancelled; retry later')
        if state is None:
            state = PDConnectionState()
            self.pool[link] = state
            state.task = asyncio.create_task(self._connect_with_deadline(conn_req, state))
        state.waiters += 1
        self._connect_requests += 1
        try:
            # A disconnected HTTP client must not cancel another client's handshake.
            await asyncio.shield(state.task)
        finally:
            state.waiters -= 1
            try:
                if state.waiters == 0 and not state.task.done():
                    state.status = PDConnectionStatus.Disconnected
                    state.task.cancel()
                    await asyncio.gather(state.task, return_exceptions=True)
            finally:
                self._connect_requests -= 1
                if state.status == PDConnectionStatus.Disconnected and self.pool.get(link) is state:
                    self.pool.pop(link)

    async def _connect_with_deadline(self, conn_req: PDConnectionMessage, state: PDConnectionState):
        link = (conn_req.p_url, conn_req.d_url)
        try:
            await asyncio.wait_for(self._connect_with_retries(conn_req), timeout=self.connection_timeout)
            if self.pool.get(link) is not state:
                raise ConnectionError('PD connection was removed while connecting')
            state.status = PDConnectionStatus.Connected
            self.reg_instance(EngineRole.Prefill, conn_req.p_url)
            self.reg_instance(EngineRole.Decode, conn_req.d_url)
        finally:
            if state.status != PDConnectionStatus.Connected and self.pool.get(link) is state:
                self.pool.pop(link)

    async def _connect_with_retries(self, conn_req: PDConnectionMessage):
        for attempt in range(self.max_retry_cnt):
            try:
                await self._connect_once(conn_req)
                return
            except Exception as exc:
                if attempt + 1 == self.max_retry_cnt:
                    raise ConnectionError('PDConnection Failure') from exc
                logger.warning(f'PD connection failure, retry cnt: {attempt + 1}: {exc}')

    async def _connect_once(self, conn_req: PDConnectionMessage):
        async def get_engine_config(server_endpoint):
            async with self.conn_sem:
                async with self.conn_sess.get(
                        get_server_api(server_endpoint, 'distserve/engine_info'),
                        timeout=self.aiotimeout,
                ) as resp:
                    resp.raise_for_status()
                    result = await resp.json()
                    return DistServeEngineConfig.model_validate_json(result)

        async def p2p_initialize(server_endpoint, init_request: DistServeInitRequest) -> DistServeInitResponse:
            async with self.conn_sem:
                async with self.conn_sess.post(
                        get_server_api(server_endpoint, 'distserve/p2p_initialize'),
                        json=init_request.model_dump(mode='json'),
                        timeout=self.aiotimeout,
                ) as resp:
                    resp.raise_for_status()
                    result = await resp.json()
                    response = DistServeInitResponse.model_validate(result)
                    if response.status != DistServeConnectionStatus.SUCCESS:
                        raise ConnectionError('PD initialization failed')
                    return response

        async def p2p_connect(server_endpoint, conn_request: DistServeConnectionRequest) -> DistServeConnectionResponse:
            async with self.conn_sem:
                async with self.conn_sess.post(
                        get_server_api(server_endpoint, 'distserve/p2p_connect'),
                        json=conn_request.model_dump(mode='json'),
                        timeout=self.aiotimeout,
                ) as resp:
                    resp.raise_for_status()
                    result = await resp.json()
                    response = DistServeConnectionResponse.model_validate(result)
                    if response.status != DistServeConnectionStatus.SUCCESS:
                        raise ConnectionError('PD connection failed')
                    return response

        logger.debug(f'{(conn_req.p_url, conn_req.d_url)} connecting...')
        # Step 1. Get Remote Engine Configuration
        prefill_engine_config = await get_engine_config(conn_req.p_url)
        decode_engine_config = await get_engine_config(conn_req.d_url)

        # Note: Only Same Parallel Configurations are supported by now
        assert prefill_engine_config.tp_size == decode_engine_config.tp_size

        # Step 2. Construct Initialize Configuration
        prefill_init_req = DistServeInitRequest(
            protocol=conn_req.protocol,
            local_engine_id=conn_req.p_url,
            local_engine_config=prefill_engine_config,
            remote_engine_id=conn_req.d_url,
            remote_engine_config=decode_engine_config,
            rdma_config=conn_req.rdma_config,
            nvlink_config=conn_req.nvlink_config,
        )
        decode_init_req = DistServeInitRequest(
            protocol=conn_req.protocol,
            local_engine_id=conn_req.d_url,
            local_engine_config=decode_engine_config,
            remote_engine_id=conn_req.p_url,
            remote_engine_config=prefill_engine_config,
            rdma_config=conn_req.rdma_config,
            nvlink_config=conn_req.nvlink_config,
        )

        prefill_init_resp = await p2p_initialize(conn_req.p_url, prefill_init_req)
        decode_init_resp = await p2p_initialize(conn_req.d_url, decode_init_req)

        # Step 3. Connection
        prefill_endpoint_conn_reqs = DistServeConnectionRequest(
            protocol=conn_req.protocol,
            remote_engine_id=conn_req.d_url,
            remote_engine_endpoint_info=decode_init_resp.engine_endpoint_info,
            remote_kvtransfer_endpoint_info=decode_init_resp.kvtransfer_endpoint_info)
        decode_endpoint_conn_reqs = DistServeConnectionRequest(
            protocol=conn_req.protocol,
            remote_engine_id=conn_req.p_url,
            remote_engine_endpoint_info=prefill_init_resp.engine_endpoint_info,
            remote_kvtransfer_endpoint_info=prefill_init_resp.kvtransfer_endpoint_info)
        await p2p_connect(conn_req.p_url, prefill_endpoint_conn_reqs)
        await p2p_connect(conn_req.d_url, decode_endpoint_conn_reqs)
        logger.debug(f'{(conn_req.p_url, conn_req.d_url)} connected')

    async def close(self):
        """Cancel handshakes and close HTTP resources at proxy shutdown."""
        self._closed = True
        tasks = [state.task for state in self.pool.values() if state.task is not None]
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        self.pool.clear()
        if self.conn_sess is not None:
            await self.conn_sess.close()

    def is_connected(self, p_url: str, d_url: str):
        link = self.pool.get((p_url, d_url), None)
        if not link:
            return False
        return link.status == PDConnectionStatus.Connected

    def drop(self, pd_key: tuple[str, str]):
        state = self.pool.pop(pd_key, None)
        if state is not None and state.task is not None and not state.task.done():
            state.task.get_loop().call_soon_threadsafe(state.task.cancel)
        left = pd_key[0]
        right = pd_key[1]

        def cache_free(server_endpoint, cache_free_request: DistServeCacheFreeRequest) -> dict:
            try:
                requests.post(get_server_api(server_endpoint, 'distserve/free_cache'),
                              json=cache_free_request.model_dump(mode='json'),
                              timeout=self.connection_timeout)
            except Exception as e:
                logger.warning(f'error cache block free {server_endpoint, cache_free_request}. ErrorMsg: {str(e)}')

        def drop_connect(server_endpoint: str, p2p_disconnect_request: DistServeDropConnectionRequest):
            try:
                requests.post(get_server_api(server_endpoint, 'distserve/p2p_drop_connect'),
                              json=p2p_disconnect_request.model_dump(mode='json'),
                              timeout=self.connection_timeout)
            except Exception as e:
                logger.warning(f'error drop connect {server_endpoint, p2p_disconnect_request}. ErrorMsg: {str(e)}')

        # trigger gc
        logger.warning('cache block gc triggered.')
        try:
            for session_id in list(self.migration_session_shelf.get((left, right), ())):
                cache_free(left, DistServeCacheFreeRequest(remote_engine_id=left, remote_session_id=session_id))
        except Exception as e:
            logger.warning(f'gc error, ErrorMsg: {str(e)}')

        # trigger p2p disconnect
        logger.warning('drop connection triggered.')
        try:
            drop_connect(left, DistServeDropConnectionRequest(engine_id=left, remote_engine_id=right))
            drop_connect(right, DistServeDropConnectionRequest(engine_id=right, remote_engine_id=left))
        except Exception as e:
            logger.warning(f'p2p disconnect error, ErrorMsg: {str(e)}')
