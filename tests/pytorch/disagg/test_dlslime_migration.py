# Copyright (c) OpenMMLab. All rights reserved.
import asyncio
import runpy
import sys
import threading
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from lmdeploy.pytorch import envs
from lmdeploy.pytorch.disagg.backend.backend import MIGRATION_BACKENDS
from lmdeploy.pytorch.disagg.conn.protocol import MigrationProtocol
from lmdeploy.pytorch.disagg.messages import AssignmentInstruct, MigrationAssignment


@pytest.mark.parametrize('async_migration', [False, True])
@pytest.mark.parametrize('transfer_fails', [False, True])
def test_dlslime_completion_wait_mode_and_errors(monkeypatch, async_migration, transfer_fails):
    # Test the Python wait contract without requiring the native DLSlime library.
    native = ModuleType('dlslime')
    native.RDMAEndpoint = object
    native.available_nic = lambda: []
    monkeypatch.setitem(sys.modules, 'dlslime', native)
    monkeypatch.setattr(MIGRATION_BACKENDS, 'register_module', lambda *a, **k: lambda cls: cls)
    module = Path(__file__).resolve().parents[3] / 'lmdeploy/pytorch/disagg/backend/dlslime.py'
    namespace = runpy.run_path(str(module))
    monkeypatch.setattr(envs, 'use_async_migration', async_migration)
    calls = []

    def wait():
        calls.append(threading.get_ident())
        if transfer_fails:
            raise RuntimeError('transfer failed')
        return 0

    def read(batch):
        assert batch == [(0, 0, 8, 4, 16)]
        return SimpleNamespace(wait=wait)

    cls = namespace['DLSlimeMigrationManagement']
    link = object.__new__(cls)
    link.endpoint = {MigrationProtocol.RDMA: SimpleNamespace(read=read)}
    assignment = MigrationAssignment(
        protocol=MigrationProtocol.RDMA, remote_engine_id='prefill',
        batch=[AssignmentInstruct(mr_key=0, source_offset=4, target_offset=8, length=16)])

    async def run():
        return await asyncio.wait_for(link.p2p_migrate(assignment), 2)

    if transfer_fails:
        with pytest.raises(RuntimeError, match='transfer failed'):
            asyncio.run(run())
    else:
        assert asyncio.run(run()) == 0
    assert len(calls) == 1
    assert (calls[0] != threading.get_ident()) == async_migration
