# Copyright (c) OpenMMLab. All rights reserved.
from typing import Tuple

import torch

from lmdeploy.messages import LongCacheConfig

from ..kernels import apply_rotary_pos_emb_longcache


def longcache_filter_kv_cache(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    past_key_value: Tuple[torch.Tensor],
    max_kv_seq_length: int,
    longcache_cfg: LongCacheConfig,
):
    """Filter key value cache with longcache.

    Args:
        query_states (torch.Tensor): query_states in shape of
            (seq_len, n_heads, head_dim)
        key_states (torch.Tensor): key_states in shape of
            (seq_len, n_kv_heads, head_dim)
        past_key_value (Tuple(torch.Tensor)): The past key value caches.
        longcache_cfg (LongCacheConfig): The longcache config.
        max_kv_seq_length (int): The max key value seq length.

    Returns:
        k_cache_select (torch.Tensor): Selected key cache in shape of
            (seq_len, n_kv_heads, head_dim)
        v_cache_select (torch.Tensor): Selected value cache in shape of
            (seq_len, n_kv_heads, head_dim)
        block_offsets (torch.Tensor): The block offsets for key value caches.
        new_kv_seq_length (torch.Tensor): The seq length of key value caches.
    """
    import einops

    _, n_heads, head_dim = query_states.shape
    _, n_kv_heads, _ = key_states.shape
    n_kv_groups = n_heads // n_kv_heads
    query_states_rg = einops.rearrange(query_states,
                                       's (n g) d -> n (g s) d',
                                       g=n_kv_groups)
    k_cache = past_key_value[0].reshape(-1, n_kv_heads, head_dim)
    v_cache = past_key_value[1].reshape(-1, n_kv_heads, head_dim)
    key_middle = k_cache[longcache_cfg.global_size:max_kv_seq_length -
                         longcache_cfg.local_size, :, :]
    scores = torch.einsum('nie,jne -> nji', query_states_rg, key_middle)
    if longcache_cfg.unique_option == 'group_unique':
        scores = scores.mean(dim=0, keepdim=True)

    if longcache_cfg.middle_size == 1:
        indices = torch.argmax(scores, dim=-2)
    else:
        indices = scores.argsort(dim=-2, descending=True)
        indices = indices[:, :longcache_cfg.middle_size, :]
    indices, counts = torch.unique(indices, return_counts=True)
    if longcache_cfg.recall_clip < indices.numel():
        indices_ids = torch.argsort(counts, descending=True)
        indices = indices[indices_ids[:longcache_cfg.recall_clip]]

    offsets = torch.arange(-longcache_cfg.span_size // 2,
                           (longcache_cfg.span_size + 1) // 2,
                           device=indices.device)
    fetch_seq_ids = (indices[:, None] + offsets[None, :]).view(-1)
    fetch_seq_ids = fetch_seq_ids.clamp(
        0, max_kv_seq_length - 1 - longcache_cfg.global_size -
        longcache_cfg.local_size)
    fetch_seq_ids = torch.unique(
        fetch_seq_ids).contiguous() + longcache_cfg.global_size
    fetch_seq_ids, _ = torch.sort(fetch_seq_ids, dim=-1)
    n_choosen = fetch_seq_ids.numel()
    n_choosen_total = (longcache_cfg.global_size + n_choosen +
                       longcache_cfg.local_size)
    block_size = past_key_value[0].shape[1]
    n_blocks = (n_choosen_total + block_size) // block_size
    k_cache_select = key_states.new_empty(
        (n_blocks * block_size, n_kv_heads, head_dim))
    v_cache_select = key_states.new_empty(
        (n_blocks * block_size, n_kv_heads, head_dim))
    k_cache_select[:longcache_cfg.global_size,
                   ...] = k_cache[:longcache_cfg.global_size, ...].clone()
    k_cache_select[longcache_cfg.global_size:longcache_cfg.global_size +
                   n_choosen, ...] = k_cache[fetch_seq_ids, ...].clone()
    k_cache_select[longcache_cfg.global_size + n_choosen:n_choosen_total,
                   ...] = k_cache[max_kv_seq_length -
                                  longcache_cfg.local_size:max_kv_seq_length,
                                  ...].clone()

    v_cache_select[:longcache_cfg.global_size,
                   ...] = v_cache[:longcache_cfg.global_size, ...]
    v_cache_select[longcache_cfg.global_size:longcache_cfg.global_size +
                   n_choosen, ...] = v_cache[fetch_seq_ids, ...]
    v_cache_select[longcache_cfg.global_size + n_choosen:n_choosen_total,
                   ...] = v_cache[max_kv_seq_length -
                                  longcache_cfg.local_size:max_kv_seq_length,
                                  ...]

    new_kv_seq_length = query_states.new_tensor([n_choosen_total],
                                                dtype=torch.long)
    block_offsets = torch.arange(n_blocks,
                                 device=query_states.device).unsqueeze(0)
    return k_cache_select, v_cache_select, block_offsets, new_kv_seq_length


def _get_rotary_emb_cos_sin(rotary_emb_obj, x, seq_len):
    if (not hasattr(rotary_emb_obj,
                    '_cos')) or (rotary_emb_obj._cos.shape[0] < seq_len):
        inv_freq_expanded = rotary_emb_obj.inv_freq[None, :]
        final_seq_len = rotary_emb_obj.max_position_embeddings * (
            (rotary_emb_obj.max_position_embeddings + seq_len) //
            rotary_emb_obj.max_position_embeddings)
        position_ids_expanded = torch.arange(final_seq_len,
                                             device=x.device)[:, None]
        # Force float32 since bfloat16 loses precision on long contexts
        # See https://github.com/huggingface/transformers/pull/29285
        freqs = position_ids_expanded.float() @ inv_freq_expanded.float()
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos().to(dtype=x.dtype)
        sin = emb.sin().to(dtype=x.dtype)
        setattr(rotary_emb_obj, '_cos', cos)
        setattr(rotary_emb_obj, '_sin', sin)
    return rotary_emb_obj._cos, rotary_emb_obj._sin


def _rotary_emb_longcache_fn(query_states, key_states, value_states,
                             n_choosen_kv, rotary_emb_obj):
    """apply rotary embedding for longcache."""

    cur_q_len = query_states.shape[0]
    device = query_states.device
    if cur_q_len > n_choosen_kv:
        position_ids_q = torch.arange(0, cur_q_len, device=device)
        position_ids_k = torch.arange(0, n_choosen_kv, device=device)
        cos, sin = _get_rotary_emb_cos_sin(rotary_emb_obj, value_states,
                                           cur_q_len)
    elif cur_q_len == n_choosen_kv:
        position_ids_q = torch.arange(0, cur_q_len, device=device)
        position_ids_k = None
        cos, sin = _get_rotary_emb_cos_sin(rotary_emb_obj, value_states,
                                           cur_q_len)
    else:
        position_ids_q = torch.arange(n_choosen_kv - cur_q_len,
                                      n_choosen_kv,
                                      device=device)
        position_ids_k = torch.arange(0, n_choosen_kv, device=device)
        cos, sin = _get_rotary_emb_cos_sin(rotary_emb_obj, value_states,
                                           n_choosen_kv)
    # inplace operation
    apply_rotary_pos_emb_longcache(query_states,
                                   key_states[:n_choosen_kv, ...],
                                   cos,
                                   sin,
                                   position_ids_q,
                                   position_ids_k=position_ids_k,
                                   q_embed=query_states,
                                   k_embed=key_states[:n_choosen_kv, ...])
    return query_states, key_states, value_states


def apply_longcache_on_kv_cache(query_states, key_states, past_key_value,
                                rotary_emb_obj, context):
    """apply longache."""
    kv_seq_length = context.kv_seq_length
    block_offsets = context.block_offsets
    max_kv_seq_length = context.max_kv_seq_length
    longcache_cfg = context.longcontext_cfg

    _, block_size, n_kv_heads, head_dim = past_key_value[0].shape
    if max_kv_seq_length > (longcache_cfg.global_size +
                            longcache_cfg.local_size):
        (k_cache_select, v_cache_select, block_offsets,
         kv_seq_length) = longcache_filter_kv_cache(query_states, key_states,
                                                    past_key_value,
                                                    max_kv_seq_length,
                                                    longcache_cfg)
        (query_states, k_cache_select,
         v_cache_select) = _rotary_emb_longcache_fn(query_states,
                                                    k_cache_select,
                                                    v_cache_select,
                                                    kv_seq_length.item(),
                                                    rotary_emb_obj)

        past_key_cache = k_cache_select.reshape(-1, block_size, n_kv_heads,
                                                head_dim).contiguous()
        past_value_cache = v_cache_select.reshape(-1, block_size, n_kv_heads,
                                                  head_dim).contiguous()
    else:
        past_key_cache = past_key_value[0][block_offsets.squeeze(),
                                           ...].clone().reshape(
                                               -1, n_kv_heads, head_dim)
        past_value_cache = past_key_value[1][block_offsets.squeeze(), ...]

        (query_states, past_key_cache,
         past_value_cache) = _rotary_emb_longcache_fn(query_states,
                                                      past_key_cache,
                                                      past_value_cache,
                                                      max_kv_seq_length,
                                                      rotary_emb_obj)
        past_key_cache = past_key_cache.reshape(-1, block_size, n_kv_heads,
                                                head_dim).contiguous()
    return past_key_cache, past_value_cache, kv_seq_length, block_offsets
