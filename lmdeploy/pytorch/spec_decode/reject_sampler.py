# Copyright (c) OpenMMLab. All rights reserved.
import enum
from typing import Optional

import torch
import triton
import triton.language as tl
from torch import LongTensor, Tensor, nn
from torch.profiler import record_function


class SamplePolicy(enum.Enum):
    """Sample policy."""

    ALL_GREEDY = enum.auto()


class RejectionSampler(nn.Module):

    def __init__(self, sample_policy: SamplePolicy = SamplePolicy.ALL_GREEDY):
        super().__init__()
        self.sample_policy = sample_policy
        assert sample_policy == SamplePolicy.ALL_GREEDY, 'only support all greedy sampling policy'

    def forward(
        self,
        target_logits: Tensor,
        draft_token_ids: LongTensor,
        bonus_token_ids: LongTensor,
        num_draft_tokens: LongTensor,
        max_spec_num: int,
        draft_probs: Optional[Tensor] = None,
    ):
        """forward
        Args:
            target_logits (Tensor): The logits of target model in shape of [num_tokens, vocab_size].
            draft_token_ids (LongTensor): The
            bonus_token_ids (LongTensor): The bonus token ids in shape of [batch_size].
            draft_probs (Tensor): The probability distribution of draft model in shape of [num_tokens, vocab_size].
                Default to ``None``.
        """
        output_token_ids, num_rejected_tokens, last_token_ids = rejection_sample(
            target_logits,
            draft_token_ids,
            bonus_token_ids,
            num_draft_tokens,
            max_spec_len=max_spec_num,
            draft_probs=draft_probs,
        )
        return output_token_ids, num_rejected_tokens, last_token_ids


@record_function('rejection_sample')
def rejection_sample(
    target_probs: Tensor,
    draft_token_ids: LongTensor,
    bonus_token_ids: LongTensor,
    num_draft_tokens: LongTensor,
    max_spec_len: int,
    draft_probs: Optional[Tensor] = None,
):
    """rejection sample
    Args:
        target_probs (Tensor):

    """
    assert draft_probs is None or draft_probs.is_contiguous()

    batch_size = bonus_token_ids.size(0)
    output_token_ids = bonus_token_ids.new_full((batch_size, max_spec_len + 1), -1)
    num_rejected_tokens = num_draft_tokens.new_zeros(batch_size)
    last_token_ids = num_draft_tokens.new_zeros(batch_size)
    cu_num_draft_tokens = num_draft_tokens.new_zeros(batch_size + 1)
    cu_num_draft_tokens[1:] = num_draft_tokens.cumsum(dim=0)
    grid = (batch_size, )
    target_argmax = target_probs.argmax(dim=-1)
    rejection_greedy_sample_kernel[grid](output_token_ids, cu_num_draft_tokens, draft_token_ids, target_argmax,
                                         bonus_token_ids, num_rejected_tokens, last_token_ids, max_spec_len)
    return output_token_ids, num_rejected_tokens, last_token_ids


# modify from vllm
@triton.jit(do_not_specialize=['max_spec_len'])
def rejection_greedy_sample_kernel(
    output_token_ids_ptr,  # [batch_size, max_spec_len + 1]
    cu_num_draft_tokens_ptr,  # [batch_size + 1]
    draft_token_ids_ptr,  # [num_tokens]
    target_argmax_ptr,  # [num_tokens]
    bonus_token_ids_ptr,  # [batch_size]
    num_rejected_tokens_ptr,  # [batch_size]
    last_token_ids_ptr,  # [batch_size]
    max_spec_len,
):
    seq_idx = tl.program_id(0)
    start_idx = tl.load(cu_num_draft_tokens_ptr + seq_idx)
    end_idx = tl.load(cu_num_draft_tokens_ptr + seq_idx + 1)
    num_draft_tokens = end_idx - start_idx

    rejected = False
    num_accept_tokens: int = 0
    last_token_id: int = -1
    for pos in range(num_draft_tokens):
        if not rejected:
            draft_token_id = tl.load(draft_token_ids_ptr + start_idx + pos)
            target_argmax_id = tl.load(target_argmax_ptr + start_idx + pos)
            tl.store(output_token_ids_ptr + seq_idx * (max_spec_len + 1) + pos, target_argmax_id)
            last_token_id = target_argmax_id.to(tl.int32)
            if draft_token_id != target_argmax_id:
                # Reject.
                rejected = True
            else:
                num_accept_tokens += 1

    tl.store(num_rejected_tokens_ptr + seq_idx, num_draft_tokens - num_accept_tokens)

    if not rejected:
        # If all tokens are accepted, append the bonus token.
        bonus_token_id = tl.load(bonus_token_ids_ptr + seq_idx)
        tl.store(output_token_ids_ptr + seq_idx * (max_spec_len + 1) + num_draft_tokens, bonus_token_id)
        last_token_id = bonus_token_id.to(tl.int32)

    tl.store(last_token_ids_ptr + seq_idx, last_token_id)


if __name__ == '__main__':
    num_seq = 3
    max_spec_len = 2

    num_draft_tokens = torch.LongTensor([2, 0, 1]).cuda()
    vocab_size = 3
    target_probs = torch.tensor([[0.5, 0.3, 0.2], [0.3, 0.5, 0.2], [0.1, 0.7, 0.2]], dtype=torch.bfloat16).cuda()
    draft_token_ids = torch.tensor([0, 1, 2], dtype=torch.long).cuda()
    bonus_token_ids = torch.tensor([2, 0, 2], dtype=torch.long).cuda()
    expected_last_token_ids = torch.tensor([2, 0, 1], dtype=torch.long).cuda()
    expected_num_rejected_tokens = torch.tensor([0, 0, 1], dtype=torch.long).cuda()
    expected_output_token_ids = torch.tensor([[0, 1, 2], [0, -1, -1], [1, -1, -1]], dtype=torch.long).cuda()

    draft_probs = None
    output_token_ids, num_rejected_tokens, last_token_ids = rejection_sample(
        target_probs,
        draft_token_ids,
        bonus_token_ids,
        num_draft_tokens,
        max_spec_len,
        draft_probs,
    )
    torch.testing.assert_close(output_token_ids, expected_output_token_ids)
    torch.testing.assert_close(num_rejected_tokens, expected_num_rejected_tokens)
    torch.testing.assert_close(last_token_ids, expected_last_token_ids)
