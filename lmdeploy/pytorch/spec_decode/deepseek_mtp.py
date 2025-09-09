# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict

import torch

from lmdeploy.utils import get_logger

from ..model_inputs import ModelInputs, SpecDecodeInputs
from .base import SPEC_PROPOSERS, BaseSpecProposer

logger = get_logger('lmdeploy')


@SPEC_PROPOSERS.register_module(name='deepseek_mtp')
class DeepseekMTP(BaseSpecProposer):

    def get_outputs(self,
                    model_outputs: Dict[str, torch.Tensor],
                    model_inputs: ModelInputs,
                    spec_inputs: SpecDecodeInputs = None):
        """Get outputs."""
        hidden_states = model_outputs['hidden_states']
        model_metas = model_outputs['model_metas']
        if not model_inputs.is_decoding:
            assert spec_inputs is not None, 'spec_inputs should be provided for prefill mode'
            if model_inputs.seq_length.size(0) == 1 and spec_inputs.num_rejected_tokens is None:
                hidden_states = hidden_states[:, -1:]
            else:
                last_token_loc = spec_inputs.last_token_indices
                hidden_states = hidden_states[:, last_token_loc]

        logits = self.get_logits(hidden_states)[0]
        draft_token_ids = logits.argmax(dim=-1, keepdim=True)
        return draft_token_ids, model_metas, hidden_states

    def prepare_inputs(self, model_inputs: ModelInputs, spec_inputs: SpecDecodeInputs):
        """Prepare inputs."""
        spec_metadata = model_inputs.spec_metadata
        input_ids = model_inputs.input_ids
        seq_length = model_inputs.seq_length
        last_token_indices = spec_inputs.last_token_indices
        # # offset by 1 token
        input_ids[:, :-1] = input_ids[:, 1:].clone()
        # # update next tokens
        input_ids[:, last_token_indices] = spec_inputs.next_token_ids
        # use new inputs
        return ModelInputs(
            input_ids=input_ids,
            seq_length=seq_length,
            max_kv_seqlen=model_inputs.max_kv_seqlen,
            max_q_seqlen=model_inputs.max_q_seqlen,
            sum_kv_seqlen=model_inputs.sum_kv_seqlen,
            history_lengths=model_inputs.history_lengths,
            block_offsets=model_inputs.block_offsets,
            num_ignored_history=model_inputs.num_ignored_history,
            is_decoding=model_inputs.is_decoding,
            target_hidden_states=spec_inputs.target_hidden_states,
            target_position_ids=spec_inputs.target_position_ids,
            spec_metadata=spec_metadata,
        )
