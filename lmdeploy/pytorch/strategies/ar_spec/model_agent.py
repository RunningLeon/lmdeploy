# Copyright (c) OpenMMLab. All rights reserved.
from dataclasses import dataclass
from typing import Any, List, Optional

import torch
from torch.profiler import record_function

from lmdeploy.pytorch.engine.logits_process import SamplingInputs
from lmdeploy.pytorch.messages import SchedulerSequence
from lmdeploy.pytorch.model_inputs import ModelInputs

from ..ar.model_agent import ARStoppingCriteria
from ..base.model_agent import ExtraInputs, ExtraOutputs, ModelAgentStrategy

SeqList = List[SchedulerSequence]


class ARSpecExtraInputs(ExtraInputs):
    """ARSpec extra inputs."""
    draft_token_ids: torch.Tensor = None
    num_rejected_tokens: torch.Tensor = None
    last_token_ids: torch.Tensor = None


class ARSpecExtraOutputs(ExtraOutputs):
    """ARSpec extra outputs."""
    draft_token_ids: torch.Tensor


@dataclass
class ARSpecStoppingCriteria(ARStoppingCriteria):
    num_appendable_ids: torch.Tensor

    @record_function('stopping_criteria')
    def step(self,
             token_ids: torch.Tensor,
             stop_words: torch.Tensor,
             inputs: Optional[ModelInputs] = None,
             extra_inputs: Optional[ARSpecExtraInputs] = None):
        """Check whether to stop generation."""
        if token_ids.ndim == 1:
            token_ids = token_ids.unsqueeze(-1)
        if token_ids.size(-1) == 1:
            return super().step(token_ids, stop_words, inputs, extra_inputs)

        mask = (self.num_appendable_ids.unsqueeze(-1) - (token_ids > -1).cumsum(dim=-1)) <= 0
        if stop_words is not None:
            token_ids_rsp = token_ids.unsqueeze(-1).repeat(1, 1, stop_words.numel())
            stop_words_rsp = stop_words.reshape(1, 1, -1)
            assert stop_words_rsp.ndim == token_ids_rsp.ndim == 3
            stop_mask = (token_ids_rsp == stop_words_rsp).any(-1)
            mask = mask ^ stop_mask
        # find the index of first `1`,  if not found, would be 0
        index = torch.argmax(mask.int(), dim=-1, keepdim=True)
        # update index of 0 to -1 if not found
        stop_pos = torch.where(index == 0, mask[:, 0:1].int() - 1, index).ravel()
        # TODO check if sync
        num_valid_tokens = (token_ids > -1).sum(dim=-1)
        num_appendable_ids = torch.clamp_max(self.num_appendable_ids - num_valid_tokens, 0)
        stopped = stop_pos == -1
        return stopped, stop_pos, ARSpecStoppingCriteria(num_appendable_ids=num_appendable_ids)


class ARSpecModelAgentStrategy(ModelAgentStrategy):

    def __init__(self, num_spec_tokens: int):
        self.num_spec_tokens = num_spec_tokens

    def slice_outputs(self, inputs: torch.Tensor, seq_length: torch.LongTensor) -> torch.Tensor:
        """Slice outputs."""
        # batch size == 1
        if len(seq_length) == 1:
            return inputs[-1:]

        if len(seq_length) == inputs.size(0):
            return inputs
        last_idx = seq_length.cumsum(-1) - 1
        return inputs[last_idx]

    def slice_extra_inputs(self, extra_inputs: ARSpecExtraInputs, seq_length: torch.LongTensor) -> ARSpecExtraInputs:
        """Slice outputs."""
        return extra_inputs

    def _step_sampling_inputs(self, sampling_inputs: SamplingInputs, next_token_ids: torch.Tensor):
        """step."""
        sampling_inputs.num_ignore_eos = sampling_inputs.num_ignore_eos - 1

        all_ids = sampling_inputs.all_ids
        if all_ids is not None:
            sampling_inputs.all_ids = torch.cat([all_ids, next_token_ids[:, None]], 1)

        guided_input_ids = sampling_inputs.guided_input_ids
        if guided_input_ids is not None:
            sampling_inputs.guided_input_ids = torch.cat([guided_input_ids, next_token_ids[:, None]], 1)

        return sampling_inputs

    def make_stopping_criteria(self, seqs: SeqList) -> ARSpecStoppingCriteria:
        """Create stopping criteria."""
        num_appendable = [seq.sampling_param.max_new_tokens - seq.num_new_tokens for seq in seqs]
        num_appendable = torch.tensor(num_appendable)
        return ARSpecStoppingCriteria(num_appendable_ids=num_appendable)

    def make_extra_inputs(self, seqs: 'SeqList') -> ExtraInputs:
        """Create extra inputs."""
        return ARSpecExtraInputs()

    def make_extra_outputs(self, extra_inputs: ARSpecExtraInputs) -> ARSpecExtraOutputs:
        """Create extra outputs."""
        return ARSpecExtraOutputs(draft_token_ids=extra_inputs.draft_token_ids)

    def update_inputs_for_next_step(self, model_inputs: 'ModelInputs', sampling_inputs: 'SamplingInputs',
                                    next_token_ids: torch.Tensor, model_metas: Any, extra_inputs: ARSpecExtraInputs,
                                    **kwARSpecgs):
        """Step next inputs."""
        model_inputs.model_metas = model_metas
        step_seqlens = model_inputs.seq_length
        step_seqlens -= extra_inputs.num_rejected_tokens
        model_inputs.step(next_token_ids, step_seqlens)
        model_inputs.input_ids[:, -self.num_spec_tokens:] = extra_inputs.draft_token_ids
        model_inputs.input_ids[:, 0:1] = extra_inputs.last_token_ids
        # TODO check this
        self._step_sampling_inputs(sampling_inputs, next_token_ids)
        return model_inputs, extra_inputs

    def post_sampling(self, inputs: 'ModelInputs', logits: torch.Tensor, next_token_ids: torch.LongTensor,
                      extra_inputs: ARSpecExtraInputs):
        """Post sampling."""
        return next_token_ids, extra_inputs
