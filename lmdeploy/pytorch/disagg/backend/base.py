# Copyright (c) OpenMMLab. All rights reserved.
import asyncio
from abc import abstractmethod

from lmdeploy.pytorch.disagg.conn.protocol import (
    DistServeInitRequest,
    DistServeKVTransferEndpointInfo,
    MigrationProtocol,
)
from lmdeploy.pytorch.disagg.messages import DistServeRegisterMRMessage, MigrationAssignment


async def run_transfer_in_executor(func, *args):
    """Offload a blocking transfer without abandoning in-flight memory
    access."""
    future = asyncio.get_running_loop().run_in_executor(None, func, *args)
    try:
        return await asyncio.shield(future)
    except asyncio.CancelledError:
        # Cancelling an asyncio waiter cannot stop a native RDMA operation.
        # Keep ownership until it completes, even if cancellation is repeated.
        while not future.done():
            try:
                await asyncio.shield(future)
            except asyncio.CancelledError:
                continue
            except Exception:
                break
        if not future.cancelled():
            future.exception()  # Consume transfer errors; preserve cancellation.
        raise


class MigrationBackendImpl:

    @abstractmethod
    def p2p_initialize(self, init_request: DistServeInitRequest):
        raise NotImplementedError

    @abstractmethod
    def register_memory_region(self, register_mr_request: DistServeRegisterMRMessage):
        raise NotImplementedError

    @abstractmethod
    def endpoint_info(self, remote_engine_id: str, protocol: MigrationProtocol):
        return NotImplementedError

    @abstractmethod
    def p2p_connect(self, remote_engine_id: str, conn_req: DistServeKVTransferEndpointInfo):
        raise NotImplementedError

    @abstractmethod
    def p2p_migrate(self, assignment: MigrationAssignment, async_op: bool = False):
        raise NotImplementedError

    @abstractmethod
    def store(self, assignment: MigrationAssignment, async_op: bool = False):
        raise NotImplementedError

    @abstractmethod
    def load(self, assignment: MigrationAssignment, async_op: bool = False):
        raise NotImplementedError
