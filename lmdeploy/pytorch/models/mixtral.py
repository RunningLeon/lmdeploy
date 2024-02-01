# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional, Tuple

import torch
import torch.distributed as dist
from torch import nn
from torch.distributed._tensor import DeviceMesh

from ..dist_utils import (colwise_parallelize_linear_fn,
                          rowwise_parallelize_linear_fn)
from ..kernels import apply_rotary_pos_emb, fill_kv_cache, paged_attention_fwd


class PatchedMixtralAttention(nn.Module):
    """Rewrite module of MixtralAttention."""

    @classmethod
    def _distribute_partition_fn(cls, mod_name: str, mod: nn.Module,
                                 device_mesh: DeviceMesh):
        """Distribution partition callback."""
        if mod_name in ['q_proj', 'k_proj', 'v_proj']:
            colwise_parallelize_linear_fn(mod,
                                          device_mesh=device_mesh,
                                          to_local=True)
        elif mod_name in ['o_proj']:
            rowwise_parallelize_linear_fn(mod,
                                          device_mesh=device_mesh,
                                          to_local=True)

    @classmethod
    def _distribute_output_fn(cls, outputs, device_mesh: DeviceMesh):
        """Distribution output hook."""
        dist.all_reduce(outputs[0])
        return outputs

    def _contiguous_batching_forward_impl(
        self,
        hidden_states: torch.Tensor,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        attention_mask: Optional[torch.Tensor] = None,
        world_size: int = 1,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor],
               Optional[Tuple[torch.Tensor]]]:
        """default rewrite."""

        context = self.context.context
        history_lengths = context.history_lengths
        kv_seq_length = context.kv_seq_length
        q_seq_length = context.seq_length
        q_start_loc = context.q_start_loc
        block_offsets = context.block_offsets

        num_heads = self.num_heads // world_size
        num_kv_heads = self.num_key_value_heads // world_size
        hidden_size = num_heads * self.head_dim

        def __qkv_proj(hidden_states):
            """qkv proj."""
            query_states = self.q_proj(hidden_states)
            key_states = self.k_proj(hidden_states)
            value_states = self.v_proj(hidden_states)

            return query_states, key_states, value_states

        def __rotary_emb_fn(query_states, key_states, value_states):
            if hasattr(self, 'rotary_emb'):
                max_seq_len = position_ids.size(-1)
                kv_seq_len = max_seq_len + max(history_lengths)
                cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
                query_states, key_states = apply_rotary_pos_emb(
                    query_states, key_states, cos, sin, position_ids,
                    getattr(context, 'position_ids_1d', None))
            return query_states, key_states, value_states

        query_states, key_states, value_states = __qkv_proj(hidden_states)

        query_states = query_states.view(-1, num_heads, self.head_dim)
        key_states = key_states.view(-1, num_kv_heads, self.head_dim)
        value_states = value_states.view(-1, num_kv_heads, self.head_dim)

        query_states, key_states, value_states = __rotary_emb_fn(
            query_states, key_states, value_states)
        # fill kv cache
        fill_kv_cache(key_states,
                      value_states,
                      past_key_value[0],
                      past_key_value[1],
                      q_start_loc,
                      q_seq_length,
                      block_offsets=block_offsets,
                      history_lengths=history_lengths,
                      context=context)
        # page attention
        attn_output = query_states
        max_seq_len = position_ids.size(-1)

        paged_attention_fwd(
            query_states,
            past_key_value[0],
            past_key_value[1],
            attn_output,
            block_offsets,
            b_start_loc=q_start_loc,
            b_seq_len=q_seq_length,
            b_kv_seq_len=kv_seq_length,
            max_input_len=max_seq_len,
        )

        attn_output = attn_output.reshape(*hidden_states.shape[:-1],
                                          hidden_size)

        attn_output = self.o_proj(attn_output)
        return attn_output, None, past_key_value

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor],
               Optional[Tuple[torch.Tensor]]]:
        """Rewrite of MistralAttention.forward."""
        world_size = 1
        if dist.is_initialized():
            world_size = dist.get_world_size()
        return self._contiguous_batching_forward_impl(
            hidden_states,
            position_ids,
            past_key_value,
            output_attentions,
            attention_mask=attention_mask,
            world_size=world_size,
        )


class PatchedMixtralBLockSparseTop2MLP(nn.Module):

    @classmethod
    def _distribute_partition_fn(cls, mod_name: str, mod: nn.Module,
                                 device_mesh: DeviceMesh):
        """Distribution partition callback."""
        if mod_name in ['w1', 'w3']:
            colwise_parallelize_linear_fn(mod,
                                          device_mesh=device_mesh,
                                          to_local=True)
        elif mod_name in ['w2']:
            rowwise_parallelize_linear_fn(mod,
                                          device_mesh=device_mesh,
                                          to_local=True)

    @classmethod
    def _distribute_output_fn(cls, outputs, device_mesh: DeviceMesh):
        """Distribution output hook."""
        dist.all_reduce(outputs)
        return outputs
