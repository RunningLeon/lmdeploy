import asyncio

import torch

from lmdeploy.pytorch.config import DistConfig
from lmdeploy.pytorch.distributed import DistContext
from lmdeploy.pytorch.spec_decode.spec_agent import SpecModelAgent, _build_draft_dist_ctx, _expand_sampling_inputs

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def test_build_draft_dist_ctx_reuses_main_ctx_for_same_tp():
    """Draft TP matching main TP should reuse the existing distributed
    context."""
    main_dist_config = DistConfig(tp=2)
    main_dist_ctx = DistContext(rank=1, dist_config=main_dist_config)
    specdecode_config = type('SpecDecodeConfig', (), {'dist_config': DistConfig(tp=2)})()

    draft_dist_ctx = _build_draft_dist_ctx(main_dist_ctx, specdecode_config)

    assert draft_dist_ctx is main_dist_ctx


def test_build_draft_dist_ctx_builds_local_ctx_for_tp_one(monkeypatch):
    """Draft TP=1 should build a local rank-0 distributed context."""
    main_dist_ctx = DistContext(rank=1, dist_config=DistConfig(tp=2))
    draft_dist_config = DistConfig()
    specdecode_config = type('SpecDecodeConfig', (), {'dist_config': draft_dist_config})()
    expected_dist_ctx = DistContext(rank=0, dist_config=draft_dist_config)
    build_calls = []

    def fake_build(rank, dist_config):
        build_calls.append((rank, dist_config))
        return expected_dist_ctx

    monkeypatch.setattr(DistContext, 'build', fake_build)

    draft_dist_ctx = _build_draft_dist_ctx(main_dist_ctx, specdecode_config)

    assert draft_dist_ctx is expected_dist_ctx
    assert build_calls == [(0, draft_dist_config)]


def test_build_draft_dist_ctx_reuses_main_ctx_for_dp_ep_main_tp():
    """Draft TP matching main attention TP should keep the global DP/EP
    world."""
    main_dist_config = DistConfig(dp=2, ep=8, dp_rank=1)
    main_dist_ctx = DistContext(rank=4, dist_config=main_dist_config)
    specdecode_config = type('SpecDecodeConfig', (), {'dist_config': main_dist_config})()

    draft_dist_ctx = _build_draft_dist_ctx(main_dist_ctx, specdecode_config)

    assert draft_dist_ctx is main_dist_ctx
    assert draft_dist_ctx.dist_config.world_size == 8
    assert draft_dist_ctx.dist_config.attn_tp == 4


def test_build_draft_dist_ctx_uses_local_world_for_dp_ep_tp_one(monkeypatch):
    """Draft TP=1 should not try to build groups over the main DP/EP world."""
    main_dist_ctx = DistContext(rank=4, dist_config=DistConfig(dp=2, ep=8, dp_rank=1))
    draft_dist_config = DistConfig()
    specdecode_config = type('SpecDecodeConfig', (), {'dist_config': draft_dist_config})()
    expected_dist_ctx = DistContext(rank=0, dist_config=draft_dist_config)
    build_calls = []

    def fake_build(rank, dist_config):
        build_calls.append((rank, dist_config.world_size, dist_config.attn_tp))
        return expected_dist_ctx

    monkeypatch.setattr(DistContext, 'build', fake_build)

    draft_dist_ctx = _build_draft_dist_ctx(main_dist_ctx, specdecode_config)

    assert draft_dist_ctx is expected_dist_ctx
    assert build_calls == [(0, 1, 1)]


def test_async_sampling_logits_returns_raw_sampled_extra_inputs(monkeypatch):
    """Sampling should not prepare draft inputs before broadcast."""
    agent = object.__new__(SpecModelAgent)
    calls = []

    async def rejection_sampling(model_inputs, extra_inputs, sampling_inputs):
        calls.append(('sampling', model_inputs, extra_inputs, sampling_inputs))
        return 'sampled_extra'

    def prepare_inputs(model_inputs, extra_inputs):
        raise AssertionError('async_sampling_logits must not prepare draft inputs')

    monkeypatch.setattr(agent, '_rejection_sampling', rejection_sampling)
    monkeypatch.setattr(agent, '_prepare_inputs_from_main', prepare_inputs)

    result = asyncio.run(agent.async_sampling_logits('main_inputs', 'target_extra', 'sampling_inputs'))

    assert result == 'sampled_extra'
    assert calls == [
        ('sampling', 'main_inputs', 'target_extra', 'sampling_inputs'),
    ]


def test_async_model_forward_prepares_draft_inputs(monkeypatch):
    """Draft forward should prepare draft inputs from raw sampled extra
    inputs."""
    agent = object.__new__(SpecModelAgent)
    calls = []

    async def fail_rejection_sampling(*args, **kwargs):
        raise AssertionError('async_model_forward must consume prepared draft inputs')

    def prepare_inputs(model_inputs, extra_inputs):
        calls.append(('prepare', model_inputs, extra_inputs))
        return 'draft_inputs', 'draft_extra'

    async def draft_forward(inputs, extra_inputs, sampling_inputs):
        calls.append(('forward', inputs, extra_inputs, sampling_inputs))
        return 'result'

    monkeypatch.setattr(agent, '_rejection_sampling', fail_rejection_sampling)
    monkeypatch.setattr(agent, '_prepare_inputs_from_main', prepare_inputs)
    monkeypatch.setattr(agent, '_async_model_forward', draft_forward)

    result = asyncio.run(agent.async_model_forward('main_inputs', 'sampled_extra', 'sampling_inputs'))

    assert result == 'result'
    assert calls == [
        ('prepare', 'main_inputs', 'sampled_extra'),
        ('forward', 'draft_inputs', 'draft_extra', 'sampling_inputs'),
    ]


def test_slice_sampling_inputs_decode():
    """Test _slice_sampling_inputs with decoding (num_tokens_per_batch > 1)."""
    from lmdeploy.pytorch.engine.logits_process import SamplingInputs
    from lmdeploy.pytorch.spec_decode.spec_agent import _slice_sampling_inputs

    batch_size = 2
    num_tokens_per_batch = 3

    temperature = torch.tensor([0.5, 1.0], device=device)
    top_k = torch.tensor([1, 10], device=device)
    random_offsets = torch.tensor([100, 200], device=device)

    sampling_inputs = SamplingInputs(
        max_top_k=10,
        top_k=top_k,
        temperature=temperature,
        random_offsets=random_offsets,
        max_num_logprobs=-1,
        batch_size=batch_size,
    )

    # First expand
    expanded = _expand_sampling_inputs(sampling_inputs, num_tokens_per_batch)
    assert expanded.batch_size == batch_size * num_tokens_per_batch
    # random_offsets should be offset by arange per batch element
    # batch 0: [100, 101, 102], batch 1: [200, 201, 202]
    expected_offsets = torch.tensor([100, 101, 102, 200, 201, 202], device=device)
    torch.testing.assert_close(expanded.random_offsets, expected_offsets)

    # Then slice back (is_last=True, takes last token per batch)
    sliced = _slice_sampling_inputs(expanded, num_tokens_per_batch)
    assert sliced.batch_size == batch_size
    torch.testing.assert_close(sliced.temperature, temperature)
    torch.testing.assert_close(sliced.top_k, top_k)
    assert sliced.max_top_k == 10
    # last token per batch: offsets [102, 202]
    torch.testing.assert_close(sliced.random_offsets, torch.tensor([102, 202], device=device))

    # Slice with is_last=False (takes tokens except the last one per batch)
    sliced_draft = _slice_sampling_inputs(expanded, num_tokens_per_batch, is_last=False)
    assert sliced_draft.batch_size == batch_size * (num_tokens_per_batch - 1)
    # drops last per batch: [100, 101, 200, 201]
    torch.testing.assert_close(sliced_draft.random_offsets, torch.tensor([100, 101, 200, 201], device=device))


def test_slice_sampling_inputs_prefill():
    """Test _slice_sampling_inputs with prefill (num_tokens_per_batch=1 returns
    same object)."""
    from lmdeploy.pytorch.engine.logits_process import SamplingInputs
    from lmdeploy.pytorch.spec_decode.spec_agent import _slice_sampling_inputs

    sampling_inputs = SamplingInputs(max_top_k=1, batch_size=2)
    result = _slice_sampling_inputs(sampling_inputs, 1)
    assert result is sampling_inputs
