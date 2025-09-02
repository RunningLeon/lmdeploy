# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict

import torch

from lmdeploy.utils import get_logger

from ..config import ModelConfig
from ..model_inputs import ModelInputs
from .base import SPEC_PROPOSERS
from .deepseek_mtp import DeepseekMTP

logger = get_logger('lmdeploy')


@SPEC_PROPOSERS.register_module(name='eagle3')
class Eagle3(DeepseekMTP):

    def build_model(self, empty_init: bool, target_model: torch.nn.Module = None):
        super().build_model(empty_init, target_model=target_model)
        self.draft_id_to_target_id = self.model.draft_id_to_target_id
        if not self.model.include_embed_tokens:
            logger.info('Using embed_tokens from target model.')
            del self.model.model.embed_tokens
            self.model.model.embed_tokens = target_model.get_input_embeddings()

    def get_target_hidden_size(self, model_config: ModelConfig):
        """Get target hidden size."""
        hf_config = self.specdecode_config.model_config.hf_config
        hidden_size = getattr(hf_config, 'target_hidden_size', hf_config.hidden_size)
        return hidden_size * 3

    def get_outputs(self, model_outputs: Dict[str, torch.Tensor], model_inputs: ModelInputs):
        """Get outputs."""
        hidden_states = model_outputs['hidden_states']
        hidden_states_prenorm = model_outputs['hidden_states_prenorm']
        model_metas = model_outputs['model_metas']
        if not model_inputs.is_decoding:
            if model_inputs.seq_length.size(0) == 1:
                hidden_states = hidden_states[:, -1:]
                hidden_states_prenorm = hidden_states_prenorm[:, -1:]
            else:
                last_token_loc = model_inputs.seq_length.cumsum(0) - 1
                hidden_states = hidden_states[:, last_token_loc]
                hidden_states_prenorm = hidden_states_prenorm[:, last_token_loc]

        logits = self.get_logits(hidden_states)[0]
        draft_token_ids = logits.argmax(dim=-1, keepdim=True)
        device = draft_token_ids.device
        dtype = draft_token_ids.dtype
        # token mapping
        if self.draft_id_to_target_id.device != device or self.draft_id_to_target_id.dtype != dtype:
            self.draft_id_to_target_id = self.draft_id_to_target_id.to(dtype=draft_token_ids.dtype,
                                                                       device=draft_token_ids.device)
        draft_token_ids = self.draft_id_to_target_id[draft_token_ids]
        return draft_token_ids, model_metas, hidden_states_prenorm
