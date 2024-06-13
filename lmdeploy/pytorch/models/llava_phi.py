# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Optional, Tuple, Union

import torch
import torch.distributed as dist
from torch import nn
from torch.distributed._tensor import DeviceMesh
from transformers.modeling_outputs import CausalLMOutputWithPast

from ..dist_utils import (colwise_parallelize_linear_fn,
                          colwise_split_parallelize_linear_fn,
                          rowwise_parallelize_linear_fn)
from ..kernels import fill_kv_cache, paged_attention_fwd


class PatchedMLP(nn.Module):

    @classmethod
    def _distribute_partition_fn(cls, mod_name: str, mod: nn.Module,
                                 device_mesh: DeviceMesh):
        """Distribution partition callback."""
        if mod_name in ['fc1']:
            colwise_parallelize_linear_fn(mod,
                                          device_mesh=device_mesh,
                                          to_local=True)
        elif mod_name in ['fc2']:
            rowwise_parallelize_linear_fn(mod,
                                          device_mesh=device_mesh,
                                          to_local=True)

    @classmethod
    def _distribute_output_fn(cls, outputs, device_mesh: DeviceMesh):
        """Distribution output hook."""
        dist.all_reduce(outputs)
        return outputs


class PatchedMHA(nn.Module):

    def _distribute_qkv_linear(self, mod: nn.Module, device_mesh: DeviceMesh):
        """distribute qkv linear."""
        sections = [
            self.n_head * self.head_dim,
            self.n_head_kv * self.head_dim,
            self.n_head_kv * self.head_dim,
        ]
        colwise_split_parallelize_linear_fn(mod, sections, device_mesh)

    def _distribute_partition_fn(self, mod_name: str, mod: nn.Module,
                                 device_mesh: DeviceMesh):
        """Distribution partition callback."""
        if mod_name in ['Wqkv']:
            self._distribute_qkv_linear(mod, device_mesh)
        elif mod_name in ['out_proj']:
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
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        world_size: int = 1,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor],
               Optional[Tuple[torch.Tensor]]]:
        """Rewrite implementation of LlamaAttention.forward.

        Add continuous batching support. Add paged attention support. TP
        support.
        """
        from einops import rearrange

        context = self.context.context
        q_start_loc = context.q_start_loc
        kv_seq_length = context.kv_seq_length
        q_seq_length = context.q_seq_length
        block_offsets = context.block_offsets
        max_q_seq_length = context.max_q_seq_length

        num_heads = self.n_head // world_size
        num_kv_heads = self.n_head_kv // world_size
        head_dim = self.head_dim
        hidden_size = num_heads * head_dim

        def __qkv_proj(hidden_states):
            """qkv proj."""
            qkv_states = self.Wqkv(hidden_states)
            qkv_states = rearrange(qkv_states,
                                   '... (three h d) -> ... three h d',
                                   three=3,
                                   d=head_dim)
            return qkv_states

        def __rotary_emb_fn(qkv_states):
            """rotary embedding func."""
            if self.rotary_dim > 0:
                qkv_states = self.rotary_emb(qkv_states)
            q, k, v = qkv_states.unbind(dim=2)
            return q, k, v

        qkv_states = __qkv_proj(hidden_states)

        # inplace rotary
        query_states, key_states, value_states = __rotary_emb_fn(qkv_states)

        query_states = query_states.view(-1, num_heads, head_dim)
        key_states = key_states.view(-1, num_kv_heads, head_dim)
        value_states = value_states.view(-1, num_kv_heads, head_dim)

        fill_kv_cache(
            key_states,
            value_states,
            past_key_value[0],
            past_key_value[1],
            q_start_loc,
            q_seq_length,
            kv_seq_length=kv_seq_length,
            max_q_seq_length=max_q_seq_length,
            block_offsets=block_offsets,
        )

        attn_output = query_states
        paged_attention_fwd(
            query_states,
            past_key_value[0],
            past_key_value[1],
            attn_output,
            block_offsets,
            q_start_loc=q_start_loc,
            q_seqlens=q_seq_length,
            kv_seqlens=kv_seq_length,
            max_seqlen=max_q_seq_length,
        )
        attn_output = attn_output.reshape(*hidden_states.shape[:-1],
                                          hidden_size)

        attn_output = self.out_proj(attn_output)

        return attn_output, None

    def forward(
        self,
        hidden_states: torch.Tensor,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        **kwargs
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor],
               Optional[Tuple[torch.Tensor]]]:
        """rewrite of forward."""
        world_size = 1
        if dist.is_initialized():
            world_size = dist.get_world_size()
        return self._contiguous_batching_forward_impl(
            hidden_states,
            past_key_values,
            world_size=world_size,
        )


class PatchedPhiModel(nn.Module):

    def forward(self,
                input_ids: torch.LongTensor,
                past_key_values: Optional[Union[torch.FloatTensor]] = None,
                select_layer: Optional[int] = None,
                **kwargs) -> torch.FloatTensor:
        """rewrite forward of PhiModel."""
        context = self.context.context
        # get inputs from context
        vision_embeddings = context.input_embeddings
        vision_embedding_indexing = context.input_embedding_indexing

        inputs_embeds = self.embd(input_ids)

        if vision_embeddings is not None and len(vision_embeddings) > 0:
            # multi-modality
            inputs_embeds[:,
                          vision_embedding_indexing, :] = vision_embeddings.to(
                              inputs_embeds)

        hidden_states = inputs_embeds

        for i, layer in enumerate(self.h):
            hidden_states, _ = layer(
                hidden_states,
                past_key_values=past_key_values[i],
            )
            if select_layer is not None:
                if i == select_layer:
                    return hidden_states
        return hidden_states


class PatchedLlavaPhiForCausalLM(nn.Module):

    def forward(
        self,
        input_ids: torch.LongTensor,
        past_key_values: Optional[Union[torch.FloatTensor]] = None,
        select_layer: Optional[int] = None,
        **kwargs,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        """rewrite forward of LlavaPhiForCausalLM."""
        if select_layer is None:
            select_layer = 15
        hidden_states = self.transformer(input_ids,
                                         past_key_values=past_key_values,
                                         select_layer=select_layer)
        return CausalLMOutputWithPast(logits=hidden_states)
