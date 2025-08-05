# Copyright (c) OpenMMLab. All rights reserved.
import torch
import triton
import triton.language as tl

from lmdeploy.utils import get_logger

from ..engine.cache_engine import CacheEngine
from ..model_inputs import ModelInputs, SpecDecodeInputs
from .base import SPEC_PROPOSERS, BaseSpecProposer, draft_model_forward

logger = get_logger('lmdeploy')


@SPEC_PROPOSERS.register_module(name='deepseek_mtp')
class DeepseekMTP(BaseSpecProposer):

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
        hidden_states = outputs['hidden_states']
        last_token_loc = model_inputs.seq_length.cumsum(0) - 1
        hidden_states = hidden_states[:, last_token_loc]
        model_metas = outputs['model_metas']
        logits = self.get_logits(hidden_states)[0]
        draft_token_ids = logits.argmax(dim=-1, keepdim=True)
        if self.num_speculative_tokens == 1:
            return draft_token_ids

        draft_tokens_li = [draft_token_ids]
        # update model_inputs as in decoding mode
        model_inputs = self.update_inputs_decoding(model_inputs, draft_token_ids.transpose(0, 1), hidden_states,
                                                   model_metas)
        num_forward_loop = self.num_speculative_tokens - 1

        for idx in range(num_forward_loop):
            # update inputs
            outputs = draft_model_forward(
                self.model,
                model_inputs,
                model_config=self.specdecode_config.model_config,
                cache_engine=cache_engine,
                stream=stream,
            )
            hidden_states = outputs['hidden_states']
            model_metas = outputs['model_metas']
            logits = self.get_logits(hidden_states)[0]
            draft_token_ids = logits.argmax(dim=-1, keepdim=True)
            # update inputs
            if idx < num_forward_loop - 1:
                model_inputs.update(draft_token_ids.transpose(0, 1))
                model_inputs.model_metas = model_metas
                model_inputs.target_hidden_states = hidden_states
                model_inputs.target_position_ids += 1

            draft_tokens_li.append(draft_token_ids)

        final_draft_token_ids = torch.cat(draft_tokens_li, dim=-1)
        return final_draft_token_ids

    def prepare_inputs(self, model_inputs: ModelInputs, spec_inputs: SpecDecodeInputs):
        """Prepare inputs."""
        spec_metadata = model_inputs.spec_metadata
        is_decoding = False
        history_lengths = model_inputs.history_lengths.clone()
        block_offsets = model_inputs.block_offsets
        num_ignored_history = model_inputs.num_ignored_history
        max_q_seqlen = model_inputs.max_q_seqlen
        max_kv_seqlen = model_inputs.max_kv_seqlen
        sum_kv_seqlen = model_inputs.sum_kv_seqlen
        model_metas = model_inputs.model_metas

        if spec_metadata.draft_token_ids is None:
            input_ids = model_inputs.input_ids
            seq_length = model_inputs.seq_length
        else:
            # select input ids
            query_lens = model_inputs.seq_length
            batch_size = query_lens.size(0)
            cum_query_lens = query_lens.new_zeros((batch_size + 1), dtype=torch.long)
            cum_qery_lens_new = query_lens.new_zeros((batch_size + 1), dtype=torch.long)
            torch.cumsum(query_lens, dim=0, out=cum_query_lens[1:])
            query_lens_new = query_lens - spec_inputs.num_rejected_tokens
            torch.cumsum(query_lens_new, dim=0, out=cum_qery_lens_new[1:])
            keep_token_indices = query_lens.new_zeros(
                model_inputs.input_ids.size(1) - spec_inputs.num_rejected_tokens.sum())
            cal_token_indices[(batch_size, )](keep_token_indices, cum_query_lens, cum_qery_lens_new, BLOCK_SIZE=1024)
            input_ids = model_inputs.input_ids[:, keep_token_indices]
            seq_length = query_lens_new

            # not update max_q_seqlen, max_kv_seqlen, sum_kv_seqlen for simplicity
            # model_inputs.max_q_seqlen = query_lens_new.max().item()
            # kv_seqlens = (query_lens_new + model_inputs.history_lengths).cpu()
            # model_inputs.max_kv_seqlen = kv_seqlens.max().item()
            # model_inputs.sum_kv_seqlen = kv_seqlens.sum().item()
            spec_inputs.target_hidden_states = spec_inputs.target_hidden_states[:, keep_token_indices]
            spec_inputs.target_position_ids = spec_inputs.target_position_ids[:, keep_token_indices]

        # offset by 1 token
        last_token_indices = seq_length.cumsum(0) - 1
        input_ids[:, :-1] = input_ids[:, 1:].clone()
        input_ids[:, last_token_indices] = spec_inputs.next_token_ids

        draft_model_inputs = ModelInputs(
            input_ids=input_ids,
            seq_length=seq_length,
            history_lengths=history_lengths,
            block_offsets=block_offsets,
            is_decoding=is_decoding,
            num_ignored_history=num_ignored_history,
            max_q_seqlen=max_q_seqlen,
            max_kv_seqlen=max_kv_seqlen,
            sum_kv_seqlen=sum_kv_seqlen,
            model_metas=model_metas,
            target_hidden_states=spec_inputs.target_hidden_states,
            target_position_ids=spec_inputs.target_position_ids,
        )

        return draft_model_inputs


@triton.jit
def cal_token_indices(
    token_indices_ptr,
    cum_query_lens_ptr,
    cum_new_query_lens_ptr,
    BLOCK_SIZE: tl.constexpr,
):
    """Calculate the token indices based on rejection sampler results."""
    pid = tl.program_id(0)

    start_pos = tl.load(cum_new_query_lens_ptr + pid)
    end_pos = tl.load(cum_new_query_lens_ptr + pid + 1)
    num_tokens = end_pos - start_pos

    index_start = tl.load(cum_query_lens_ptr + pid)

    num_blocks = tl.cdiv(num_tokens, BLOCK_SIZE)
    for i in tl.range(num_blocks):
        offset = i * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        tl.store(
            token_indices_ptr + start_pos + offset,
            index_start + offset,
            mask=offset < num_tokens,
        )
