# Copyright (c) OpenMMLab. All rights reserved.
import asyncio
import runpy
import threading
from types import SimpleNamespace

import pytest

from lmdeploy.pytorch import envs
from lmdeploy.pytorch.disagg.backend.mooncake import MooncakeBackend, MooncakeMigrationManagement
from lmdeploy.pytorch.disagg.conn.protocol import MigrationProtocol
from lmdeploy.pytorch.disagg.messages import AssignmentInstruct, MigrationAssignment


@pytest.mark.parametrize('async_migration', [False, True])
@pytest.mark.parametrize('transfer_result', [0, -1])
def test_mooncake_transfer_completes_or_propagates_failure(monkeypatch, async_migration, transfer_result):
    """Exercise both backend layers and the actual _migrate return contract."""
    monkeypatch.setattr(envs, 'use_async_migration', async_migration)
    calls = []

    def transfer(*args):
        calls.append((args, threading.get_ident()))
        return transfer_result

    link = object.__new__(MooncakeMigrationManagement)
    link.local_engine_id = 'decode'
    link.remote_engine_id = 'prefill'
    link.remote_url = 'prefill-transfer'
    link.local_kv_table = {'0': {'addr': 100, 'length': 64}}
    link.remote_kv_table = {'0': {'addr': 200, 'length': 64}}
    link.engine = SimpleNamespace(transfer_sync_read=transfer)
    backend = MooncakeBackend()
    backend.links['prefill'] = link
    assignment = MigrationAssignment(
        protocol=MigrationProtocol.RDMA,
        remote_engine_id='prefill',
        batch=[AssignmentInstruct(mr_key=0, source_offset=4, target_offset=8, length=16)],
    )

    async def run():
        # In the async success case the old code waited on an uncompleted Future.
        return await asyncio.wait_for(backend.p2p_migrate(assignment), timeout=2)

    if transfer_result:
        with pytest.raises(RuntimeError, match='Failed to perform sync transfer: -1'):
            asyncio.run(run())
    else:
        assert asyncio.run(run()) is None
    assert len(calls) == 1
    assert calls[0][0] == ('prefill-transfer', 104, 208, 16)
    assert (calls[0][1] != threading.get_ident()) == async_migration


@pytest.mark.parametrize('value, expected', [(None, False), ('0', False), ('1', True)])
def test_async_migration_environment_flag(monkeypatch, value, expected):
    if value is None:
        monkeypatch.delenv('LMDEPLOY_USE_ASYNC_MIGRATION', raising=False)
    else:
        monkeypatch.setenv('LMDEPLOY_USE_ASYNC_MIGRATION', value)
    # Read a fresh env snapshot without reloading registered migration backends.
    snapshot = runpy.run_path(envs.__file__)
    assert snapshot['use_async_migration'] is expected
    if value is not None:
        assert snapshot['_ENVS']['LMDEPLOY_USE_ASYNC_MIGRATION'] == value


@pytest.mark.parametrize('transfer_fails', [False, True])
def test_async_transfer_cancellation_waits_for_native_completion(transfer_fails):
    from lmdeploy.pytorch.disagg.backend.base import run_transfer_in_executor

    async def run():
        started = asyncio.Event()
        release = threading.Event()
        completed = threading.Event()
        loop = asyncio.get_running_loop()

        def transfer():
            loop.call_soon_threadsafe(started.set)
            assert release.wait(5)
            completed.set()
            if transfer_fails:
                raise RuntimeError('transfer failed after cancellation')

        caller = asyncio.create_task(run_transfer_in_executor(transfer))
        try:
            await asyncio.wait_for(started.wait(), 2)
            for _ in range(2):
                caller.cancel()
                await asyncio.sleep(0.01)
                assert not caller.done()
                assert not completed.is_set()
        finally:
            release.set()
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(caller, 2)
        assert completed.is_set()

    asyncio.run(run())
