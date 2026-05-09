# Copyright (c) OpenMMLab. All rights reserved.
from types import SimpleNamespace

import pytest

from lmdeploy.cli.utils import get_speculative_config
from lmdeploy.messages import PytorchEngineConfig, SpeculativeConfig
from lmdeploy.pytorch.config import CacheConfig, DistConfig, SpecDecodeConfig
from lmdeploy.pytorch.engine.config_builder import ConfigBuilder


def test_speculative_config_tp_defaults_to_one():
    spec_config = SpeculativeConfig(method='qwen3_5_mtp')

    assert spec_config.tp == 1


def test_cli_speculative_tp_maps_to_config():
    args = SimpleNamespace(
        speculative_algorithm='qwen3_5_mtp',
        speculative_draft_model='draft-model',
        speculative_num_draft_tokens=4,
        speculative_tp=8,
    )

    spec_config = get_speculative_config(args)

    assert spec_config.tp == 8


def test_build_specdecode_config_accepts_main_tp_for_draft(monkeypatch):
    engine_config = PytorchEngineConfig(tp=8)
    main_dist_config = DistConfig.from_engine_config(engine_config)
    cache_config = CacheConfig(max_batches=1, block_size=64, num_cpu_blocks=0, num_gpu_blocks=0)
    spec_config = SpeculativeConfig(method='qwen3_5_mtp', tp=8)
    captured = {}

    def fake_from_config(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(dist_config=kwargs['dist_config'])

    monkeypatch.setattr(SpecDecodeConfig, 'from_config', staticmethod(fake_from_config))

    specdecode_config = ConfigBuilder.build_specdecode_config(
        'target-model',
        spec_config,
        engine_config,
        cache_config,
        trust_remote_code=False,
        dist_config=main_dist_config,
    )

    assert captured['dist_config'].attn_tp == 8
    assert specdecode_config.dist_config.attn_tp == 8


def test_build_specdecode_config_keeps_dp_ep_world_for_main_tp_draft(monkeypatch):
    engine_config = PytorchEngineConfig(tp=8, dp=2, ep=8, dp_rank=1)
    main_dist_config = DistConfig.from_engine_config(engine_config)
    cache_config = CacheConfig(max_batches=1, block_size=64, num_cpu_blocks=0, num_gpu_blocks=0)
    spec_config = SpeculativeConfig(method='qwen3_5_mtp', tp=main_dist_config.attn_tp)
    captured = {}

    def fake_from_config(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(dist_config=kwargs['dist_config'])

    monkeypatch.setattr(SpecDecodeConfig, 'from_config', staticmethod(fake_from_config))

    specdecode_config = ConfigBuilder.build_specdecode_config(
        'target-model',
        spec_config,
        engine_config,
        cache_config,
        trust_remote_code=False,
        dist_config=main_dist_config,
    )

    assert main_dist_config.world_size == 8
    assert main_dist_config.attn_tp == 4
    assert captured['dist_config'].world_size == 8
    assert captured['dist_config'].attn_tp == 4
    assert captured['dist_config'].dp == 2
    assert captured['dist_config'].ep == 8
    assert specdecode_config.dist_config.world_size == 8


def test_build_specdecode_config_rejects_intermediate_draft_tp(monkeypatch):
    engine_config = PytorchEngineConfig(tp=8)
    main_dist_config = DistConfig.from_engine_config(engine_config)
    cache_config = CacheConfig(max_batches=1, block_size=64, num_cpu_blocks=0, num_gpu_blocks=0)
    spec_config = SpeculativeConfig(method='qwen3_5_mtp', tp=2)

    with pytest.raises(ValueError, match='speculative tp'):
        ConfigBuilder.build_specdecode_config(
            'target-model',
            spec_config,
            engine_config,
            cache_config,
            trust_remote_code=False,
            dist_config=main_dist_config,
        )
