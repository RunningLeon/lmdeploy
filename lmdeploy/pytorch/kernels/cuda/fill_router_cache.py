# Copyright (c) OpenMMLab. All rights reserved.
import triton
import triton.language as tl
from torch import Tensor


@triton.jit
def _fill_router_cache_kernel(
    ExpertIds,
    RouterCache,
    QStartLoc,
    QSeqLens,
    KVSeqLens,
    BlockOffsets,
    is_decoding: tl.constexpr,
    topk: tl.constexpr,
    stride_ess,
    stride_esk: tl.constexpr,
    stride_rcn: tl.constexpr,
    stride_rcb: tl.constexpr,
    stride_rck: tl.constexpr,
    stride_boff,
    BLOCK: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """Fill router cache kernel."""
    batch_id = tl.program_id(1)
    block_id = tl.program_id(0)

    q_startloc = tl.load(QStartLoc + batch_id)
    q_seqlen = tl.load(QSeqLens + batch_id)
    kv_seqlen = tl.load(KVSeqLens + batch_id)
    history_seqlen = kv_seqlen - q_seqlen

    kv_block_id = history_seqlen // BLOCK + block_id

    if kv_seqlen <= 0:
        return

    if kv_block_id * BLOCK >= kv_seqlen:
        return

    if is_decoding:
        page_offs = tl.full((1, ), history_seqlen % BLOCK, dtype=tl.int32)
        kv_mask = tl.full((1, ), 1, dtype=tl.int1)
        q_offs = tl.full((1, ), q_startloc, dtype=tl.int32)
    else:
        page_offs = tl.arange(0, BLOCK)
        kv_offs = kv_block_id * BLOCK + page_offs
        kv_mask = (kv_offs >= history_seqlen) & (kv_offs < kv_seqlen)
        token_off = q_startloc + kv_block_id * BLOCK - history_seqlen
        q_offs = token_off + page_offs

    block_off = tl.load(BlockOffsets + batch_id * stride_boff + kv_block_id)

    d_off = tl.arange(0, BLOCK_D)
    mask_ks = kv_mask[:, None]
    mask_kc = mask_ks & (d_off[None, :] < topk)

    exp_ptrs = ExpertIds + q_offs[:, None] * stride_ess + d_off[None, :] * stride_esk
    cache_ptr = RouterCache + block_off * stride_rcn
    cache_ptrs = cache_ptr + page_offs[:, None] * stride_rcb + d_off[None, :] * stride_rck

    exp_ids = tl.load(exp_ptrs, mask=mask_ks)
    tl.store(cache_ptrs, exp_ids, mask=mask_kc)


def fill_router_cache(expert_ids: Tensor, router_cache: Tensor, q_start_loc: Tensor, q_seq_length: Tensor,
                      kv_seq_length: Tensor, max_q_seq_length: int, block_offsets: Tensor):
    """Fill expert ids to router cache for moe reuse."""
    expert_ids = expert_ids.to(dtype=router_cache.dtype)
    block_offsets = block_offsets.contiguous()
    batch_size = block_offsets.size(0)
    _, block_size, topk = router_cache.shape

    if max_q_seq_length == 1:
        max_num_blocks = 1
    else:
        max_num_blocks = triton.cdiv(max_q_seq_length, block_size) + 1

    BLOCK = block_size
    BLOCK_D = triton.next_power_of_2(topk)
    grid = (max_num_blocks, batch_size)

    is_decoding = max_num_blocks == 1

    _fill_router_cache_kernel[grid](
        expert_ids,
        router_cache,
        q_start_loc,
        q_seq_length,
        kv_seq_length,
        block_offsets,
        is_decoding=is_decoding,
        topk=topk,
        stride_ess=expert_ids.stride(0),
        stride_esk=expert_ids.stride(1),
        stride_rcn=router_cache.stride(0),
        stride_rcb=router_cache.stride(1),
        stride_rck=router_cache.stride(2),
        stride_boff=block_offsets.stride(0),
        BLOCK=BLOCK,
        BLOCK_D=BLOCK_D,
        num_warps=4,
        num_stages=3,
    )


if __name__ == '__main__':
    import torch
    topk = 7
    router_cache = torch.full((513, 64, topk), 0, dtype=torch.int16).cuda()
    expert_ids = torch.randint(10, (6, topk), dtype=torch.long).cuda()
    q_seq_lens = torch.tensor([1, 2, 3], dtype=torch.long).cuda()
    kv_seq_lens = torch.tensor([1, 2, 3], dtype=torch.long).cuda()
    q_start_loc = q_seq_lens.cumsum(0) - q_seq_lens
    block_offsets = torch.tensor([[0, 1], [2, 3], [4, 5]], dtype=torch.long).cuda()
    max_q_seq_length = 3
    fill_router_cache(expert_ids, router_cache, q_start_loc, q_seq_lens, kv_seq_lens, max_q_seq_length, block_offsets)

    for idx in range(q_seq_lens.size(0)):
        num_q = q_seq_lens[idx]
        experts_gt = expert_ids[q_start_loc[idx]:q_start_loc[idx] + q_seq_lens[idx]]
        experts_res = router_cache[block_offsets[idx]].flatten(0, 1)[:num_q]
        torch.testing.assert_close(experts_gt, experts_res.to(torch.long))
