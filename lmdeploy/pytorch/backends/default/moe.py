# Copyright (c) OpenMMLab. All rights reserved.
from typing import Any

import torch

from ...kernels.cuda.fill_router_cache import fill_router_cache
from ..moe import SoftmaxTopKBuilder, SoftmaxTopKImpl


class DefaultSoftmaxTopKImpl(SoftmaxTopKImpl):
    """RMS norm implementation api."""

    def __init__(self, top_k: int, dim: int = -1):
        self.top_k = top_k
        self.dim = dim

    def forward(
        self,
        x: torch.Tensor,
        router_cache: torch.Tensor = None,
        attn_metadata: Any = None,
    ):
        """forward."""
        routing_weights = torch.softmax(x, dim=self.dim, dtype=torch.float32)
        topk_weights, topk_ids = torch.topk(routing_weights, self.top_k, dim=self.dim)

        # fill router cache
        if router_cache is not None and attn_metadata is not None:
            batch_size, _ = attn_metadata.block_offsets.shape
            if attn_metadata.is_decoding:
                max_q_seqlen = x.size(0) // batch_size
            else:
                max_q_seqlen = x.size(0)
            fill_router_cache(
                topk_ids,
                router_cache,
                q_start_loc=attn_metadata.q_start_loc,
                q_seq_length=attn_metadata.q_seqlens,
                kv_seq_length=attn_metadata.kv_seqlens,
                max_q_seq_length=max_q_seqlen,
                block_offsets=attn_metadata.block_offsets,
            )
        return topk_weights, topk_ids


class DefaultSoftmaxTopKBuilder(SoftmaxTopKBuilder):
    """RMS norm implementation builder."""

    @staticmethod
    def build(top_k: int, dim: int = -1):
        """build."""
        return DefaultSoftmaxTopKImpl(top_k, dim)
