# Copyright (c) OpenMMLab. All rights reserved.
import asyncio
from contextlib import contextmanager
from types import SimpleNamespace

import torch

from lmdeploy.pytorch.disagg.config import EngineRole
from lmdeploy.pytorch.engine.model_agent import agent as agent_module
from lmdeploy.pytorch.engine.model_agent.agent import BaseModelAgent
from lmdeploy.pytorch.strategies.ar_spec import model_agent as ar_spec_agent_module
from lmdeploy.pytorch.strategies.ar_spec.model_agent import ARSpecExtraInputs, ARSpecModelAgentStrategy


def test_non_output_rank_runs_draft_after_next_token_broadcast():
    """Non-output TP ranks should run draft forward after sampled inputs are
    broadcast."""
    agent = object.__new__(BaseModelAgent)
    agent.rank = 1
    agent.dist_ctx = object()
    calls = []

    class DummySpecAgent:

        def is_enabled(self):
            return True

        async def async_sampling_logits(self, *args, **kwargs):
            raise AssertionError('non-output ranks must not run target/rejection sampling')

        async def async_model_forward(self, inputs, extra_inputs, sampling_inputs):
            assert calls and calls[-1][0] == 'broadcast_next_exit'
            assert inputs == 'main_inputs'
            assert extra_inputs.broadcasted
            calls.append(('draft_forward', inputs, extra_inputs, sampling_inputs))
            return extra_inputs

    class DummyStrategy:

        def make_dummy_next_token(self, inputs, logits, extra_inputs):
            calls.append(('make_dummy_next_token', inputs, logits, extra_inputs))
            extra_inputs.next_token_ids = torch.tensor([7])
            return extra_inputs.next_token_ids, extra_inputs

        def make_extra_outputs(self, extra_inputs):
            calls.append(('make_extra_outputs', extra_inputs))
            return 'extra_outputs'

        def broadcast_output_draft_token_ids(self, extra_inputs, dist_ctx):
            calls.append(('broadcast_draft', extra_inputs, dist_ctx))

    @contextmanager
    def fake_broadcast_next_token(next_token_ids, extra_inputs, enable=True):
        calls.append(('broadcast_next', next_token_ids, extra_inputs, enable))
        extra_inputs.broadcasted = True
        yield
        calls.append(('broadcast_next_exit', next_token_ids, extra_inputs, enable))

    agent.spec_agent = DummySpecAgent()
    agent.agent_strategy = DummyStrategy()
    agent._broadcast_next_token = fake_broadcast_next_token

    result = asyncio.run(
        agent._step_postprocess_without_output(
                inputs='main_inputs',
                last_logits=torch.zeros(1, 1),
                extra_inputs=SimpleNamespace(),
                sampling_inputs='sampling_inputs',
                need_broadcast_next=True,
            ))

    assert result[1].tolist() == [7]
    assert result[3] == 'extra_outputs'
    assert [call[0] for call in calls] == [
        'make_dummy_next_token',
        'broadcast_next',
        'broadcast_next_exit',
        'draft_forward',
        'broadcast_draft',
        'make_extra_outputs',
    ]


def test_output_postprocess_samples_spec_inputs_before_broadcast():
    """Output rank postprocess should keep spec sampling aligned with non-spec
    sampling."""
    agent = object.__new__(BaseModelAgent)
    agent.rank = 0
    agent.dist_ctx = object()
    agent._pytorch_engine_config = SimpleNamespace(thread_safe=False)
    calls = []

    class DummySpecAgent:

        def is_enabled(self):
            return True

        async def async_sampling_logits(self, inputs, extra_inputs, sampling_inputs):
            calls.append(('spec_sampling', inputs, extra_inputs, sampling_inputs))
            return SimpleNamespace(
                next_token_ids=torch.tensor([9]),
                output_token_ids=torch.tensor([[9, 10]]),
                logprobs='logprobs',
                source='sampled',
            )

        async def async_model_forward(self, inputs, extra_inputs, sampling_inputs):
            assert calls and calls[-1][0] == 'broadcast_next_exit'
            assert inputs == 'main_inputs'
            assert extra_inputs.source == 'sampled'
            calls.append(('draft_forward', inputs, extra_inputs, sampling_inputs))
            return extra_inputs

    class DummyStrategy:

        def make_extra_outputs(self, extra_inputs):
            calls.append(('make_extra_outputs', extra_inputs))
            return 'extra_outputs'

        def broadcast_output_draft_token_ids(self, extra_inputs, dist_ctx):
            calls.append(('broadcast_draft', extra_inputs, dist_ctx))

    class DummyStoppingCriteria:

        def step(self, next_token_ids, stop_words, inputs=None, extra_inputs=None):
            calls.append(('stopping', next_token_ids, stop_words, inputs, extra_inputs))
            return torch.tensor([False]), None, self

    @contextmanager
    def fake_broadcast_next_token(next_token_ids, extra_inputs, enable=True):
        calls.append(('broadcast_next', next_token_ids, extra_inputs, enable))
        assert extra_inputs.source == 'sampled'
        yield
        calls.append(('broadcast_next_exit', next_token_ids, extra_inputs, enable))

    def fake_push_output(output):
        calls.append(('push_output', output))

    agent.spec_agent = DummySpecAgent()
    agent.agent_strategy = DummyStrategy()
    agent._broadcast_next_token = fake_broadcast_next_token
    agent._push_output = fake_push_output

    result = asyncio.run(
        agent._step_postprocess_with_output(
            last_logits=torch.zeros(1, 1),
            logits=torch.zeros(1, 1),
            inputs='main_inputs',
            sampling_inputs=SimpleNamespace(stop_words='stop_words'),
            stopping_criteria=DummyStoppingCriteria(),
            model_metas='model_metas',
            need_broadcast_next=True,
            extra_inputs='target_extra',
        ))

    assert result[4].tolist() == [9]
    assert [call[0] for call in calls] == [
        'spec_sampling',
        'broadcast_next',
        'broadcast_next_exit',
        'draft_forward',
        'broadcast_draft',
        'stopping',
        'make_extra_outputs',
        'push_output',
    ]


def test_async_step_delegates_spec_sampling_to_output_postprocess(monkeypatch):
    """Spec sampling should stay inside the output postprocess helper."""
    agent = object.__new__(BaseModelAgent)
    agent.rank = 0
    agent.need_output = True
    agent.cache_engine = SimpleNamespace()
    agent.cache_config = SimpleNamespace(role=EngineRole.Prefill)
    agent._prev_chunk_output = None
    calls = []

    class DummySpecAgent:

        def is_enabled(self):
            return True

        async def async_sampling_logits(self, *args, **kwargs):
            raise AssertionError('_async_step should not run spec sampling directly')

    class DummyStrategy:

        def slice_outputs(self, logits, seq_length):
            calls.append(('slice_outputs', logits, seq_length))
            return torch.zeros(1, 4)

        def slice_extra_inputs(self, extra_inputs, inputs, output):
            calls.append(('slice_extra_inputs', extra_inputs, inputs, output))
            return extra_inputs

    class DummySamplingInputs:

        def get_delta(self):
            calls.append(('sampling_delta', ))
            return 'sampling_delta'

    async def fake_model_forward(inputs, return_logits, extra_inputs):
        calls.append(('model_forward', inputs, return_logits, extra_inputs))
        return {
            'logits': [torch.zeros(1, 4)],
            'seq_length': inputs.seq_length,
            'model_metas': 'model_metas',
        }

    async def fake_postprocess_with_output(last_logits, logits, inputs, sampling_inputs, stopping_criteria, model_metas,
                                           need_broadcast_next, return_logits=False, all_routed_experts=None,
                                           extra_inputs=None):
        calls.append(('postprocess_with_output', extra_inputs))
        assert extra_inputs == 'main_extra'
        assert calls[-2][0] == 'slice_extra_inputs'
        return inputs, extra_inputs, stopping_criteria, 'extra_outputs', torch.tensor([3])

    fake_dist_ctx = SimpleNamespace(dist_config=SimpleNamespace(attn_tp=2, dp=1))
    monkeypatch.setattr(agent_module, 'get_dist_manager',
                        lambda: SimpleNamespace(current_context=lambda: fake_dist_ctx))

    agent.spec_agent = DummySpecAgent()
    agent.agent_strategy = DummyStrategy()
    agent._async_model_forward = fake_model_forward
    agent._step_postprocess_with_output = fake_postprocess_with_output

    inputs = SimpleNamespace(
        is_dummy=False,
        is_first_chunk=True,
        is_chunk=False,
        is_decoding=True,
        seq_length=torch.tensor([1]),
        input_ids=torch.tensor([1]),
    )

    asyncio.run(
        agent._async_step(
            inputs=inputs,
            sampling_inputs=DummySamplingInputs(),
            stopping_criteria='stopping_criteria',
            extra_inputs='main_extra',
        ))

    assert [call[0] for call in calls] == [
        'model_forward',
        'slice_outputs',
        'slice_extra_inputs',
        'postprocess_with_output',
        'sampling_delta',
    ]


def test_non_output_rank_runs_draft_forward_after_broadcast():
    """Non-output TP ranks should run draft forward after sampled fields
    sync."""
    agent = object.__new__(BaseModelAgent)
    agent.rank = 1
    agent.dist_ctx = object()
    calls = []

    class DummySpecAgent:

        def is_enabled(self):
            return True

        async def async_model_forward(self, inputs, extra_inputs, sampling_inputs):
            assert calls and calls[-1][0] == 'broadcast_next_exit'
            calls.append(('draft_forward', inputs, extra_inputs, sampling_inputs))
            assert inputs == 'main_inputs'
            return extra_inputs

    class DummyStrategy:

        def make_dummy_next_token(self, inputs, logits, extra_inputs):
            calls.append(('make_dummy_next_token', inputs, logits, extra_inputs))
            extra_inputs.next_token_ids = torch.tensor([7])
            return extra_inputs.next_token_ids, extra_inputs

        def make_extra_outputs(self, extra_inputs):
            calls.append(('make_extra_outputs', extra_inputs))
            return 'extra_outputs'

        def broadcast_output_draft_token_ids(self, extra_inputs, dist_ctx):
            calls.append(('broadcast_draft', extra_inputs, dist_ctx))

    @contextmanager
    def fake_broadcast_next_token(next_token_ids, extra_inputs, enable=True):
        calls.append(('broadcast_next', next_token_ids, extra_inputs, enable))
        yield
        calls.append(('broadcast_next_exit', next_token_ids, extra_inputs, enable))

    agent.spec_agent = DummySpecAgent()
    agent.agent_strategy = DummyStrategy()
    agent._broadcast_next_token = fake_broadcast_next_token

    result = asyncio.run(
        agent._step_postprocess_without_output(
            inputs='main_inputs',
            last_logits=torch.zeros(1, 1),
            extra_inputs=SimpleNamespace(),
            sampling_inputs='sampling_inputs',
            need_broadcast_next=True,
        ))

    assert result[1].tolist() == [7]
    assert [call[0] for call in calls] == [
        'make_dummy_next_token',
        'broadcast_next',
        'broadcast_next_exit',
        'draft_forward',
        'broadcast_draft',
        'make_extra_outputs',
    ]


def test_ar_spec_next_token_broadcast_uses_predraft_fields(monkeypatch):
    """AR spec broadcast should sync sampled fields before draft forward."""
    strategy = ARSpecModelAgentStrategy(num_spec_tokens=3)
    calls = []

    class DummyHandle:

        def __init__(self, name):
            self.name = name

        def wait(self):
            calls.append(('wait', self.name))

    def fake_get_global_rank(group, rank):
        calls.append(('get_global_rank', group, rank))
        return 0

    def fake_broadcast(tensor, src, group, async_op=False):
        calls.append(('broadcast', tensor, src, group, async_op))
        assert tensor is not None
        return DummyHandle(f'tensor_{len(calls)}')

    monkeypatch.setattr(ar_spec_agent_module.dist, 'get_global_rank', fake_get_global_rank)
    monkeypatch.setattr(ar_spec_agent_module.dist, 'broadcast', fake_broadcast)

    next_token_ids = torch.tensor([5])
    extra_inputs = ARSpecExtraInputs(
        next_token_ids=next_token_ids,
        last_token_indices=torch.tensor([2]),
        num_rejected_tokens=torch.tensor([0]),
        output_token_ids=torch.tensor([[5, 6, -1, -1]]),
        output_draft_token_ids=None,
    )
    dist_ctx = SimpleNamespace(attn_tp_group=SimpleNamespace(gpu_group='tp_group'))

    with strategy.broadcast_next_token(next_token_ids, extra_inputs, dist_ctx):
        calls.append(('inside_context', ))

    broadcast_tensors = [call[1] for call in calls if call[0] == 'broadcast']
    assert broadcast_tensors == [
        next_token_ids,
        extra_inputs.last_token_indices,
        extra_inputs.num_rejected_tokens,
        extra_inputs.output_token_ids,
    ]
    assert ('inside_context', ) in calls
    assert len([call for call in calls if call[0] == 'wait']) == 4


def test_ar_spec_broadcasts_output_draft_token_ids_after_draft(monkeypatch):
    """AR spec should sync draft tokens after draft forward."""
    strategy = ARSpecModelAgentStrategy(num_spec_tokens=3)
    calls = []

    class DummyHandle:

        def wait(self):
            calls.append(('wait', ))

    def fake_get_global_rank(group, rank):
        calls.append(('get_global_rank', group, rank))
        return 0

    def fake_broadcast(tensor, src, group, async_op=False):
        calls.append(('broadcast', tensor, src, group, async_op))
        assert tensor is extra_inputs.output_draft_token_ids
        return DummyHandle()

    monkeypatch.setattr(ar_spec_agent_module.dist, 'get_global_rank', fake_get_global_rank)
    monkeypatch.setattr(ar_spec_agent_module.dist, 'broadcast', fake_broadcast)

    extra_inputs = ARSpecExtraInputs(output_draft_token_ids=torch.tensor([[1, 2, 3]]))
    dist_ctx = SimpleNamespace(attn_tp_group=SimpleNamespace(gpu_group='tp_group'))

    strategy.broadcast_output_draft_token_ids(extra_inputs, dist_ctx)

    assert calls == [
        ('get_global_rank', 'tp_group', 0),
        ('broadcast', extra_inputs.output_draft_token_ids, 0, 'tp_group', True),
        ('wait', ),
    ]
