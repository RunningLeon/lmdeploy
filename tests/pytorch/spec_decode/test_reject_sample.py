import torch

from lmdeploy.pytorch.engine.logits_process import SamplingInputs
from lmdeploy.pytorch.spec_decode.reject_sampler import (
    PLACEHOLDER_TOKEN_ID,
    _extract_outputs,
    _seeded_exponential_kernel,
    _seeded_uniform_kernel,
    rejection_greedy_sample_kernel,
    rejection_sample,
    sample_recovered_tokens_kernel,
    torch_greedy_rejection_sample,
)

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def _make_peaked_logits(token_ids_2d, vocab_size):
    """Build logits where argmax(dim=-1) == token_ids_2d.

    token_ids_2d: list[list[int]] or Tensor [batch, num_spec]
    """
    if isinstance(token_ids_2d, torch.Tensor):
        token_ids_2d = token_ids_2d.tolist()
    batch_size = len(token_ids_2d)
    num_spec = len(token_ids_2d[0])
    logits = torch.full((batch_size, num_spec, vocab_size), -100.0, device=device)
    for b in range(batch_size):
        for s in range(num_spec):
            logits[b, s, token_ids_2d[b][s]] = 100.0
    return logits


def _extract_valid(output_token_ids):
    """Return list-of-lists of non-placeholder token ids."""
    out = []
    for row in output_token_ids:
        out.append(row[row != PLACEHOLDER_TOKEN_ID].tolist())
    return out


class TestRejectSample:
    """Tests for rejection_sample and torch_greedy_rejection_sample.

    Uses hand-crafted token ids so the expected output is obvious.
    """

    # ----- greedy: rejection_sample -----

    def test_greedy_all_match(self):
        """All draft == target -> accept all + bonus."""
        target_logits = _make_peaked_logits([[10, 20, 30], [10, 20, 30]], 64)
        draft = torch.tensor([[10, 20, 30], [10, 20, 30]], dtype=torch.long, device=device)
        bonus = torch.tensor([99, 88], dtype=torch.long, device=device)
        si = SamplingInputs(max_top_k=1)

        out, rej, last = rejection_sample(target_logits, draft, bonus, sampling_inputs=si)

        assert out.shape == (2, 4)
        assert (rej == 0).all()
        assert _extract_valid(out) == [[10, 20, 30, 99], [10, 20, 30, 88]]
        assert last.tolist() == [99, 88]

    def test_greedy_first_mismatch(self):
        """Mismatch at position 0 -> keep only target[0]."""
        target_logits = _make_peaked_logits([[10, 20, 30], [10, 20, 30]], 64)
        draft = torch.tensor([[11, 20, 30], [11, 20, 30]], dtype=torch.long, device=device)
        bonus = torch.tensor([99, 88], dtype=torch.long, device=device)
        si = SamplingInputs(max_top_k=1)

        out, rej, last = rejection_sample(target_logits, draft, bonus, sampling_inputs=si)

        assert (rej == 3).all()
        assert _extract_valid(out) == [[10], [10]]
        assert last.tolist() == [10, 10]

    def test_greedy_middle_mismatch(self):
        """Mismatch at position 2 -> keep target[0:3], no bonus."""
        target_logits = _make_peaked_logits([[10, 20, 30], [10, 20, 30]], 64)
        draft = torch.tensor([[10, 20, 31], [10, 20, 31]], dtype=torch.long, device=device)
        bonus = torch.tensor([99, 88], dtype=torch.long, device=device)
        si = SamplingInputs(max_top_k=1)

        out, rej, last = rejection_sample(target_logits, draft, bonus, sampling_inputs=si)

        assert (rej == 1).all()
        assert _extract_valid(out) == [[10, 20, 30], [10, 20, 30]]
        assert last.tolist() == [30, 30]

    def test_greedy_mixed_mismatch_positions(self):
        """Each row has a different first-mismatch position.

        Row 0: all match       -> [10, 20, 30, 99]
        Row 1: mismatch at 0   -> [10]
        Row 2: mismatch at 1   -> [10, 20]
        Row 3: mismatch at 2   -> [10, 20, 30]
        """
        target_logits = _make_peaked_logits([[10, 20, 30]] * 4, 64)
        draft = torch.tensor([
            [10, 20, 30],
            [11, 20, 30],
            [10, 21, 30],
            [10, 20, 31],
        ],
                             dtype=torch.long,
                             device=device)
        bonus = torch.tensor([99, 88, 77, 66], dtype=torch.long, device=device)
        si = SamplingInputs(max_top_k=1)

        out, rej, last = rejection_sample(target_logits, draft, bonus, sampling_inputs=si)

        assert rej.tolist() == [0, 3, 2, 1]
        assert _extract_valid(out) == [
            [10, 20, 30, 99],
            [10],
            [10, 20],
            [10, 20, 30],
        ]
        assert last.tolist() == [99, 10, 20, 30]

    def test_greedy_all_rejected(self):
        """Every draft wrong -> only first target token kept."""
        target_logits = _make_peaked_logits([[10, 20, 30], [10, 20, 30]], 64)
        draft = torch.tensor([[11, 21, 31], [11, 21, 31]], dtype=torch.long, device=device)
        bonus = torch.tensor([99, 88], dtype=torch.long, device=device)
        si = SamplingInputs(max_top_k=1)

        out, rej, last = rejection_sample(target_logits, draft, bonus, sampling_inputs=si)

        assert (rej == 3).all()
        assert _extract_valid(out) == [[10], [10]]
        assert last.tolist() == [10, 10]

    # ----- greedy: compare rejection_sample vs torch_greedy_rejection_sample -----

    def test_greedy_triton_vs_torch_all_match(self):
        """Both implementations agree when all drafts match."""
        target_logits = _make_peaked_logits([[10, 20, 30]] * 4, 64)
        draft = torch.tensor([[10, 20, 30]] * 4, dtype=torch.long, device=device)
        bonus = torch.tensor([99, 88, 77, 66], dtype=torch.long, device=device)
        si = SamplingInputs(max_top_k=1)

        t_out, t_rej, t_last = rejection_sample(target_logits, draft, bonus, sampling_inputs=si)
        p_out, p_rej, p_last = torch_greedy_rejection_sample(target_logits, draft, bonus, sampling_inputs=si)

        assert torch.equal(t_out, p_out)
        assert torch.equal(t_rej, p_rej)
        assert torch.equal(t_last, p_last)

    def test_greedy_triton_vs_torch_mixed(self):
        """Both implementations agree with mixed mismatch positions."""
        target_logits = _make_peaked_logits([[10, 20, 30]] * 4, 64)
        draft = torch.tensor(
            [
                [10, 20, 30],  # all match
                [11, 20, 30],  # pos 0
                [10, 21, 30],  # pos 1
                [10, 20, 31],  # pos 2
            ],
            dtype=torch.long,
            device=device)
        bonus = torch.tensor([99, 88, 77, 66], dtype=torch.long, device=device)
        si = SamplingInputs(max_top_k=1)

        t_out, t_rej, t_last = rejection_sample(target_logits, draft, bonus, sampling_inputs=si)
        p_out, p_rej, p_last = torch_greedy_rejection_sample(target_logits, draft, bonus, sampling_inputs=si)

        assert torch.equal(t_out, p_out), f'output mismatch: triton={t_out}, torch={p_out}'
        assert torch.equal(t_rej, p_rej), f'num_rejected: triton={t_rej}, torch={p_rej}'
        assert torch.equal(t_last, p_last), f'last_token: triton={t_last}, torch={p_last}'

    def test_greedy_triton_vs_torch_large_batch(self):
        """Both implementations agree with large batch + random mismatches."""
        batch_size, num_spec, vocab = 32, 6, 128
        target_logits = torch.randn(batch_size, num_spec, vocab, device=device)
        target_argmax = target_logits.argmax(dim=-1)
        draft = target_argmax.clone()

        torch.manual_seed(42)
        flip = torch.rand(batch_size, num_spec) < 0.3
        for i in range(batch_size):
            for j in range(num_spec):
                if flip[i, j]:
                    draft[i, j] = (target_argmax[i, j] + 1) % vocab

        bonus = torch.randint(0, vocab, (batch_size, ), device=device)
        si = SamplingInputs(max_top_k=1)

        t_out, t_rej, t_last = rejection_sample(target_logits, draft, bonus, sampling_inputs=si)
        p_out, p_rej, p_last = torch_greedy_rejection_sample(target_logits, draft, bonus, sampling_inputs=si)

        assert torch.equal(t_out, p_out)
        assert torch.equal(t_rej, p_rej)
        assert torch.equal(t_last, p_last)

    # ----- mixed greedy/random -----

    def test_mixed_batch(self):
        """Mixed batch: greedy (top_k=1) and random sequences."""
        target_logits = _make_peaked_logits([[5, 6, 7], [5, 6, 7], [5, 6, 7], [5, 6, 7]], 32)
        draft = target_logits.argmax(dim=-1)
        bonus = torch.tensor([99, 88, 77, 66], dtype=torch.long, device=device)

        top_k = torch.tensor([1, 10, 1, 10], device=device)
        si = SamplingInputs(max_top_k=10, top_k=top_k)

        out, rej, last = rejection_sample(target_logits, draft, bonus, sampling_inputs=si)

        # Greedy rows (0, 2) must accept all since draft == target
        assert rej[0] == 0
        assert rej[2] == 0
        assert last[0] == 99
        assert last[2] == 77

    # ----- random: no draft_probs (MTP/n-gram) -----

    def test_random_no_draft_probs(self):
        """Random sampling with draft_probs=None (MTP/n-gram mode)."""
        target_logits = _make_peaked_logits([[5, 6, 7], [5, 6, 7]], 32)
        draft = torch.tensor([[5, 6, 7], [5, 6, 7]], dtype=torch.long, device=device)
        bonus = torch.tensor([99, 88], dtype=torch.long, device=device)

        top_k = torch.tensor([10, 10], device=device)
        si = SamplingInputs(max_top_k=10, top_k=top_k)

        out, rej, last = rejection_sample(target_logits, draft, bonus, sampling_inputs=si, draft_probs=None)

        assert out.shape == (2, 4)
        assert (last >= 0).all()

    # ----- random: all reject (peaked target, wrong draft) -----

    def test_random_all_reject(self):
        """Random: draft picks wrong token, peaked target -> reject at pos 0."""
        vocab = 32
        target_logits = _make_peaked_logits([[5, 6, 7], [5, 6, 7]], vocab)
        draft = torch.tensor([[1, 1, 1], [1, 1, 1]], dtype=torch.long, device=device)
        draft_probs = torch.zeros(2, 3, vocab, device=device)
        draft_probs[:, :, 1] = 1.0
        bonus = torch.tensor([99, 88], dtype=torch.long, device=device)

        top_k = torch.tensor([10, 10], device=device)
        si = SamplingInputs(max_top_k=10, top_k=top_k)

        out, rej, last = rejection_sample(target_logits, draft, bonus, sampling_inputs=si, draft_probs=draft_probs)

        # Reject at pos 0 -> 1 recovered token per row
        valid = _extract_valid(out)
        for i in range(2):
            assert len(valid[i]) == 1, f'row {i}: expected 1 token, got {valid[i]}'

    # ----- _extract_outputs -----

    def test_extract_outputs(self):
        """Test _extract_outputs helper function."""
        output_token_ids = torch.tensor([
            [5, 6, 7, 8],  # all valid
            [10, 11, -1, -1],  # 2 valid
        ])
        out, rej, last = _extract_outputs(output_token_ids, num_spec_tokens=3)
        assert rej.tolist() == [0, 2]
        assert last.tolist() == [8, 11]


# ---------------------------------------------------------------------------
# Triton kernel tests (direct kernel invocations)
# ---------------------------------------------------------------------------


class TestTritonKernels:

    def test_greedy_kernel_mixed(self):
        """rejection_greedy_sample_kernel with mixed match/mismatch rows.

        Row 0: all match       -> [10, 20, 30, 40, 50]
        Row 1: mismatch at 2   -> [10, 20, 30, -1, -1]
        Row 2: mismatch at 0   -> [10, -1, -1, -1, -1]
        """
        batch_size, num_spec = 3, 4
        target_argmax = torch.tensor([[10, 20, 30, 40]] * 3, dtype=torch.long, device=device)
        draft = torch.tensor([
            [10, 20, 30, 40],
            [10, 20, 99, 40],
            [99, 20, 30, 40],
        ], dtype=torch.long, device=device)
        bonus = torch.tensor([50, 51, 52], dtype=torch.long, device=device)
        out = torch.full((batch_size, num_spec + 1), PLACEHOLDER_TOKEN_ID, dtype=torch.long, device=device)

        rejection_greedy_sample_kernel[(batch_size, )](out, draft, target_argmax, bonus, None, num_spec)

        assert out[0].tolist() == [10, 20, 30, 40, 50]
        assert out[1].tolist() == [10, 20, 30, PLACEHOLDER_TOKEN_ID, PLACEHOLDER_TOKEN_ID]
        assert out[2].tolist() == [
            10, PLACEHOLDER_TOKEN_ID, PLACEHOLDER_TOKEN_ID, PLACEHOLDER_TOKEN_ID, PLACEHOLDER_TOKEN_ID
        ]

    def test_greedy_kernel_with_is_greedy_mask(self):
        """rejection_greedy_sample_kernel skips non-greedy rows.

        Row 0: greedy, all match -> [10, 20, 30, 50]
        Row 1: not greedy        -> all placeholder (untouched)
        """
        batch_size, num_spec = 2, 3
        target_argmax = torch.tensor([[10, 20, 30]] * 2, dtype=torch.long, device=device)
        draft = target_argmax.clone()
        bonus = torch.tensor([50, 51], dtype=torch.long, device=device)
        out = torch.full((batch_size, num_spec + 1), PLACEHOLDER_TOKEN_ID, dtype=torch.long, device=device)
        is_greedy = torch.tensor([True, False], dtype=torch.bool, device=device)

        rejection_greedy_sample_kernel[(batch_size, )](out, draft, target_argmax, bonus, is_greedy, num_spec)

        assert out[0].tolist() == [10, 20, 30, 50]
        assert (out[1] == PLACEHOLDER_TOKEN_ID).all()

    def test_recovered_tokens_with_draft_probs(self):
        """sample_recovered_tokens_kernel: recovered token = argmax of
        max(0, target - draft) * inv_q.

        target = [0, 0, 0, 0.9, 0, 0.1, 0, 0]
        draft  = [0, 0, 0, 0.9, 0, 0,   0, 0]
        diff   = [0, 0, 0, 0,   0, 0.1, 0, 0]  -> token 5
        """
        batch_size, num_spec, vocab = 2, 2, 8
        target_probs = torch.zeros(batch_size, num_spec, vocab, device=device, dtype=torch.float32)
        target_probs[:, :, 3] = 0.9
        target_probs[:, :, 5] = 0.1

        draft_probs = torch.zeros_like(target_probs)
        draft_probs[:, :, 3] = 0.9

        draft_ids = torch.full((batch_size, num_spec), 3, dtype=torch.long, device=device)
        inv_q = torch.ones(batch_size, vocab, device=device, dtype=torch.float32)
        recovered = torch.empty(batch_size, num_spec, dtype=torch.long, device=device)

        sample_recovered_tokens_kernel[(batch_size, num_spec)](recovered,
                                                               draft_ids,
                                                               draft_probs,
                                                               target_probs,
                                                               inv_q,
                                                               num_spec,
                                                               vocab,
                                                               8,
                                                               NO_DRAFT_PROBS=False)

        assert (recovered == 5).all()

    def test_recovered_tokens_no_draft_probs(self):
        """sample_recovered_tokens_kernel with NO_DRAFT_PROBS: draft token
        is masked out, recovered = argmax of remaining target probs.

        target   = [0, 0, 0.6, 0, 0, 0.3, 0, 0.1]
        draft_id = 2 (masked out)
        remaining= [0, 0, 0,   0, 0, 0.3, 0, 0.1]  -> token 5
        """
        batch_size, num_spec, vocab = 2, 2, 8
        target_probs = torch.zeros(batch_size, num_spec, vocab, device=device, dtype=torch.float32)
        target_probs[:, :, 2] = 0.6
        target_probs[:, :, 5] = 0.3
        target_probs[:, :, 7] = 0.1

        draft_ids = torch.full((batch_size, num_spec), 2, dtype=torch.long, device=device)
        inv_q = torch.ones(batch_size, vocab, device=device, dtype=torch.float32)
        recovered = torch.empty(batch_size, num_spec, dtype=torch.long, device=device)

        sample_recovered_tokens_kernel[(batch_size, num_spec)](recovered,
                                                               draft_ids,
                                                               None,
                                                               target_probs,
                                                               inv_q,
                                                               num_spec,
                                                               vocab,
                                                               8,
                                                               NO_DRAFT_PROBS=True)

        assert (recovered == 5).all()


def _make_expanded_sampling_inputs(batch_size, num_spec, seeds_base, base_offsets, top_k_val=10):
    """Build a SamplingInputs in the expanded format that spec_agent passes to
    the rejection sampler: shape [batch_size * num_spec] with seeds repeated
    and offsets shifted by position index.
    """
    seeds_base = torch.as_tensor(seeds_base, dtype=torch.long, device=device)
    base_offsets = torch.as_tensor(base_offsets, dtype=torch.long, device=device)
    arange = torch.arange(num_spec, device=device)

    seeds = seeds_base.repeat_interleave(num_spec)  # [b0]*S, [b1]*S, ...
    offsets = (base_offsets.unsqueeze(1) + arange.unsqueeze(0)).flatten()  # base[b] + pos
    top_k = torch.full((batch_size * num_spec, ), top_k_val, dtype=torch.long, device=device)

    return SamplingInputs(
        max_top_k=top_k_val,
        top_k=top_k,
        random_seeds=seeds,
        random_offsets=offsets,
        batch_size=batch_size * num_spec,
    )


class TestSeededRngKernels:
    """Direct tests for _seeded_uniform_kernel and
    _seeded_exponential_kernel."""

    def test_uniform_kernel_range(self):
        """All output values must be in (0, 1)."""
        if device == 'cpu':
            return
        batch_size, num_spec = 4, 3
        seeds = torch.randint(1, 0xFFFFFFFF, (batch_size, ), dtype=torch.long, device=device)
        offsets = torch.zeros(batch_size, dtype=torch.long, device=device)
        out = torch.empty(batch_size, num_spec, device=device, dtype=torch.float32)

        _seeded_uniform_kernel[(batch_size, num_spec)](seeds, offsets, out, num_spec)

        assert (out > 0).all(), 'uniform kernel produced 0.0'
        assert (out < 1).all(), 'uniform kernel produced >= 1.0'

    def test_uniform_kernel_reproducible(self):
        """Same seed/offset must produce identical values on repeated calls."""
        if device == 'cpu':
            return
        batch_size, num_spec = 4, 3
        seeds = torch.tensor([10, 20, 30, 40], dtype=torch.long, device=device)
        offsets = torch.tensor([100, 200, 300, 400], dtype=torch.long, device=device)

        out1 = torch.empty(batch_size, num_spec, device=device, dtype=torch.float32)
        out2 = torch.empty(batch_size, num_spec, device=device, dtype=torch.float32)
        _seeded_uniform_kernel[(batch_size, num_spec)](seeds, offsets, out1, num_spec)
        _seeded_uniform_kernel[(batch_size, num_spec)](seeds, offsets, out2, num_spec)

        assert torch.equal(out1, out2)

    def test_uniform_kernel_different_offsets(self):
        """Different offsets for the same seed must produce different
        values."""
        if device == 'cpu':
            return
        batch_size, num_spec = 4, 3
        seeds = torch.ones(batch_size, dtype=torch.long, device=device) * 42
        offsets_a = torch.tensor([0, 0, 0, 0], dtype=torch.long, device=device)
        offsets_b = torch.tensor([1000, 1000, 1000, 1000], dtype=torch.long, device=device)

        out_a = torch.empty(batch_size, num_spec, device=device, dtype=torch.float32)
        out_b = torch.empty(batch_size, num_spec, device=device, dtype=torch.float32)
        _seeded_uniform_kernel[(batch_size, num_spec)](seeds, offsets_a, out_a, num_spec)
        _seeded_uniform_kernel[(batch_size, num_spec)](seeds, offsets_b, out_b, num_spec)

        assert not torch.equal(out_a, out_b)

    def test_exponential_kernel_positive(self):
        """All output values must be strictly positive (exponential
        distribution)."""
        if device == 'cpu':
            return
        batch_size, vocab = 4, 256
        num_spec = 3
        seeds = torch.randint(1, 0xFFFFFFFF, (batch_size, ), dtype=torch.long, device=device)
        offsets = torch.zeros(batch_size, dtype=torch.long, device=device)
        out = torch.empty(batch_size, vocab, device=device, dtype=torch.float32)

        _seeded_exponential_kernel[(batch_size, )](seeds, offsets, out, num_spec, vocab, 128)

        assert (out > 0).all(), 'exponential kernel produced non-positive value'

    def test_exponential_kernel_reproducible(self):
        """Same seed/offset must produce identical values on repeated calls."""
        if device == 'cpu':
            return
        batch_size, vocab, num_spec = 4, 256, 3
        seeds = torch.tensor([10, 20, 30, 40], dtype=torch.long, device=device)
        offsets = torch.tensor([100, 200, 300, 400], dtype=torch.long, device=device)

        out1 = torch.empty(batch_size, vocab, device=device, dtype=torch.float32)
        out2 = torch.empty(batch_size, vocab, device=device, dtype=torch.float32)
        _seeded_exponential_kernel[(batch_size, )](seeds, offsets, out1, num_spec, vocab, 128)
        _seeded_exponential_kernel[(batch_size, )](seeds, offsets, out2, num_spec, vocab, 128)

        assert torch.equal(out1, out2)

    def test_exponential_kernel_different_seeds(self):
        """Different seeds must produce different exponential variates."""
        if device == 'cpu':
            return
        batch_size, vocab, num_spec = 1, 256, 3
        offsets = torch.zeros(batch_size, dtype=torch.long, device=device)

        out_a = torch.empty(batch_size, vocab, device=device, dtype=torch.float32)
        out_b = torch.empty(batch_size, vocab, device=device, dtype=torch.float32)
        seeds_a = torch.tensor([1], dtype=torch.long, device=device)
        seeds_b = torch.tensor([9999], dtype=torch.long, device=device)
        _seeded_exponential_kernel[(batch_size, )](seeds_a, offsets, out_a, num_spec, vocab, 128)
        _seeded_exponential_kernel[(batch_size, )](seeds_b, offsets, out_b, num_spec, vocab, 128)

        assert not torch.equal(out_a, out_b)

    def test_uniform_exponential_offset_no_overlap(self):
        """Uniform and exponential kernels with the same seed/base_offset must
        draw from non-overlapping offset ranges (no shared randomness)."""
        if device == 'cpu':
            return
        batch_size, num_spec, vocab = 2, 3, 64
        seeds = torch.tensor([42, 42], dtype=torch.long, device=device)
        base_offsets = torch.tensor([0, 0], dtype=torch.long, device=device)

        u_out = torch.empty(batch_size, num_spec, device=device, dtype=torch.float32)
        e_out = torch.empty(batch_size, vocab, device=device, dtype=torch.float32)
        _seeded_uniform_kernel[(batch_size, num_spec)](seeds, base_offsets, u_out, num_spec)
        _seeded_exponential_kernel[(batch_size, )](seeds, base_offsets, e_out, num_spec, vocab, 64)

        # The exponential kernel uses offsets base + num_spec + v; the uniform kernel
        # uses base + pos (0..num_spec-1). Verify the distributions look independent:
        # reciprocal of exponential = inv_q used in Gumbel; compare that to uniform.
        inv_q = e_out[:, :num_spec].reciprocal()
        # They should not be equal (offset streams are disjoint)
        assert not torch.allclose(u_out, inv_q)


class TestSeededRejectionSample:
    """Tests that rejection_sample is reproducible under the same RNG
    seeds/offsets."""

    def _make_random_inputs(self, batch_size, num_spec, vocab, seeds_base, base_offsets):
        target_logits = torch.randn(batch_size, num_spec, vocab, device=device)
        draft = torch.randint(0, vocab, (batch_size, num_spec), device=device)
        bonus = torch.randint(0, vocab, (batch_size, ), device=device)
        si = _make_expanded_sampling_inputs(batch_size, num_spec, seeds_base, base_offsets)
        return target_logits, draft, bonus, si

    def test_reproducible_same_seeds(self):
        """Two calls with identical sampling_inputs seeds/offsets must produce
        identical rejection sampling output."""
        if device == 'cpu':
            return
        batch_size, num_spec, vocab = 4, 3, 64
        seeds_base = [111, 222, 333, 444]
        base_offsets = [0, 0, 0, 0]

        torch.manual_seed(0)
        logits, draft, bonus, si = self._make_random_inputs(batch_size, num_spec, vocab, seeds_base, base_offsets)

        out1, rej1, last1 = rejection_sample(logits, draft, bonus, sampling_inputs=si)
        out2, rej2, last2 = rejection_sample(logits, draft, bonus, sampling_inputs=si)

        assert torch.equal(out1, out2), 'output_token_ids not reproducible'
        assert torch.equal(rej1, rej2), 'num_rejected_tokens not reproducible'
        assert torch.equal(last1, last2), 'last_token_ids not reproducible'

    def test_different_seeds_differ(self):
        """Two calls with different seeds should (almost certainly) differ."""
        if device == 'cpu':
            return
        batch_size, num_spec, vocab = 4, 3, 64

        torch.manual_seed(1)
        logits = torch.randn(batch_size, num_spec, vocab, device=device)
        # Use uniform target to make random sampling non-trivial
        logits = torch.zeros_like(logits)
        draft = torch.randint(0, vocab, (batch_size, num_spec), device=device)
        bonus = torch.randint(0, vocab, (batch_size, ), device=device)

        si_a = _make_expanded_sampling_inputs(batch_size, num_spec, [1, 2, 3, 4], [0, 0, 0, 0])
        si_b = _make_expanded_sampling_inputs(batch_size, num_spec, [9999, 8888, 7777, 6666], [0, 0, 0, 0])

        out_a, _, _ = rejection_sample(logits, draft, bonus, sampling_inputs=si_a)
        out_b, _, _ = rejection_sample(logits, draft, bonus, sampling_inputs=si_b)

        assert not torch.equal(out_a, out_b), 'different seeds produced identical output'

    def test_seeds_none_fallback(self):
        """rejection_sample must run without error when random_seeds is None
        (falls back to global RNG)."""
        batch_size, num_spec, vocab = 2, 3, 32
        logits = torch.randn(batch_size, num_spec, vocab, device=device)
        draft = torch.randint(0, vocab, (batch_size, num_spec), device=device)
        bonus = torch.randint(0, vocab, (batch_size, ), device=device)
        si = SamplingInputs(max_top_k=10, top_k=torch.full((batch_size, ), 10, dtype=torch.long, device=device))
        # random_seeds and random_offsets default to None

        out, rej, last = rejection_sample(logits, draft, bonus, sampling_inputs=si)

        assert out.shape == (batch_size, num_spec + 1)
        assert rej.shape == (batch_size, )
        assert (last >= 0).all()

    def test_different_offsets_differ(self):
        """Same seed but different base offsets should produce different
        output."""
        if device == 'cpu':
            return
        batch_size, num_spec, vocab = 4, 3, 64
        logits = torch.zeros(batch_size, num_spec, vocab, device=device)
        draft = torch.randint(0, vocab, (batch_size, num_spec), device=device)
        bonus = torch.randint(0, vocab, (batch_size, ), device=device)

        seeds = [42, 42, 42, 42]
        si_a = _make_expanded_sampling_inputs(batch_size, num_spec, seeds, [0, 0, 0, 0])
        si_b = _make_expanded_sampling_inputs(batch_size, num_spec, seeds, [1000, 1000, 1000, 1000])

        out_a, _, _ = rejection_sample(logits, draft, bonus, sampling_inputs=si_a)
        out_b, _, _ = rejection_sample(logits, draft, bonus, sampling_inputs=si_b)

        assert not torch.equal(out_a, out_b), 'different offsets produced identical output'
