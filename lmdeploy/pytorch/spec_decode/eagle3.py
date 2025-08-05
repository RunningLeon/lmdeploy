# Copyright (c) OpenMMLab. All rights reserved.
import torch

from lmdeploy.utils import get_logger

from ..engine.cache_engine import CacheEngine
from ..model_inputs import ModelInputs
from .base import SPEC_PROPOSERS, draft_model_forward
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

    def propose(
        self,
        model_inputs: ModelInputs,
        cache_engine: CacheEngine = None,
        stream: torch.cuda.Stream = None,
    ):
        outputs = draft_model_forward(self.model,
                                      model_inputs,
                                      model_config=self.specdecode_config.model_config,
                                      cache_engine=cache_engine,
                                      stream=stream)
        last_hidden_states = outputs['hidden_states']
        hidden_states = outputs['hidden_states_prenorm']
        last_token_loc = model_inputs.seq_length.cumsum(0) - 1
        hidden_states = hidden_states[:, last_token_loc]
        last_hidden_states = last_hidden_states[:, last_token_loc]
        model_metas = outputs['model_metas']
        logits = self.get_logits(last_hidden_states)[0]
        draft_token_ids = logits.argmax(dim=-1, keepdim=True)
        device = draft_token_ids.device
        dtype = draft_token_ids.dtype
        # token mapping
        if self.draft_id_to_target_id.device != device or self.draft_id_to_target_id.dtype != dtype:
            self.draft_id_to_target_id = self.draft_id_to_target_id.to(dtype=draft_token_ids.dtype,
                                                                       device=draft_token_ids.device)
        draft_token_ids = self.draft_id_to_target_id[draft_token_ids]
        if self.num_speculative_tokens == 1:
            return draft_token_ids

        draft_tokens_li = [draft_token_ids]

        # update model_inputs as in decoding mode
        model_inputs = self.update_inputs_decoding(model_inputs, draft_token_ids.transpose(0, 1), hidden_states,
                                                   model_metas)

        num_forward_loop = self.num_speculative_tokens - 1
        for idx in range(num_forward_loop):
            outputs = draft_model_forward(
                self.model,
                model_inputs,
                model_config=self.specdecode_config.model_config,
                cache_engine=cache_engine,
                stream=stream,
            )
            last_hidden_states = outputs['hidden_states']
            hidden_states = outputs['hidden_states_prenorm']
            model_metas = outputs['model_metas']

            logits = self.get_logits(last_hidden_states)[0]
            draft_token_ids = logits.argmax(dim=-1, keepdim=True)
            draft_token_ids = self.draft_id_to_target_id[draft_token_ids]
            # update inputs
            if idx < num_forward_loop - 1:
                model_inputs.update(draft_token_ids.transpose(0, 1))
                model_inputs.model_metas = model_metas
                model_inputs.target_hidden_states = hidden_states
                model_inputs.target_position_ids += 1

            draft_tokens_li.append(draft_token_ids)

        final_draft_token_ids = torch.cat(draft_tokens_li, dim=-1)
        return final_draft_token_ids
