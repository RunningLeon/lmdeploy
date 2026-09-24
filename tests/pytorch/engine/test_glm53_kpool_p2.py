# Copyright (c) OpenMMLab. All rights reserved.
"""Storage and rollback oracles independent of generated-answer accuracy."""
import pytest
import torch

from lmdeploy.pytorch.kernels.cuda.kpool import read_raw_tail, write_raw_ring
from lmdeploy.pytorch.nn.kpool import (
    kpool_packed_cache_views,
    kpool_pooled_block_offsets,
    kpool_read_packed_cache,
    kpool_write_packed_cache,
)

DEVICES = ['cpu'] + (['cuda'] if torch.cuda.is_available() else [])


@pytest.mark.parametrize('device', DEVICES)
@pytest.mark.parametrize('depth', [0, 1, 2, 3, 4, 7])
def test_raw_ring_all_acceptance_prefixes(device, depth):
    torch.manual_seed(4)
    cap = depth + 5
    # Deliberately strided state-slot views, as returned for a model layer.
    backing_k = torch.randn(4, 3, cap, 128, device=device, dtype=torch.bfloat16)
    backing_s = torch.randn_like(backing_k)
    keys, scores = backing_k[:, 1], backing_s[:, 1]
    pristine = (backing_k.clone(), backing_s.clone())
    sid = torch.tensor([2, -1], device=device)
    for h in [0, 1, 2, 3, 7, 63, 64, 65, 255, 256, 257, 258, 259]:
        raw_k = torch.randn(h + depth + 2, 128, device=device, dtype=torch.bfloat16)
        raw_s = torch.randn_like(raw_k)
        def write(start, n):
            history = torch.tensor([start, 0], device=device)
            lengths = torch.tensor([n, 0], device=device)
            starts = torch.tensor([0, n], device=device)
            write_raw_ring(keys, scores, raw_k[start:start+n], raw_s[start:start+n],
                           sid, history, lengths, starts)
        write(0, h)  # long prefill, wrap and retain only the last capacity rows
        saved = keys.clone(), scores.clone()
        for accepted in range(depth + 2):
            keys.copy_(saved[0])
            scores.copy_(saved[1])
            write(h, depth + 1)
            # Covers accepted-prefix 0..Q, including the exact H=258 -> 259 case.
            end = h + accepted
            history = torch.tensor([end, 0], device=device)
            got = read_raw_tail(keys, scores, sid, history)
            for out, raw in zip(got, (raw_k, raw_s)):
                n = end % 4
                torch.testing.assert_close(out[0, :n], raw[end-n:end], rtol=0, atol=0)
                assert torch.count_nonzero(out[0, n:]) == 0
                assert torch.count_nonzero(out[1]) == 0
            # Repeated rejection/revisit overwrites the proposal rather than advancing trial end.
            for _ in range(3):
                write(end, 1)
                end += 1
                if end > h + depth + 1:
                    break
                for out, raw in zip(read_raw_tail(keys, scores, sid, torch.tensor([end, 0], device=device)),
                                    (raw_k, raw_s)):
                    torch.testing.assert_close(out[0, :end % 4], raw[end-end % 4:end], rtol=0, atol=0)
        for value, initial in zip((backing_k, backing_s), pristine):
            torch.testing.assert_close(value[:, 0], initial[:, 0], rtol=0, atol=0)
            torch.testing.assert_close(value[0], initial[0], rtol=0, atol=0)


@pytest.mark.parametrize('device', DEVICES)
def test_compact_pages_fragmented_roundtrip(device):
    cache = torch.zeros((19, 16, 1, 132), dtype=torch.uint8, device=device)
    table = torch.tensor([7, 2, 11, 4, 17], device=device)
    groups = torch.arange(73, device=device)
    values = (torch.arange(73*128, device=device).reshape(73, 128) % 17).to(torch.float8_e4m3fn)
    scales = torch.arange(1, 74, device=device).float()[:, None]
    kpool_write_packed_cache(cache, table, groups, values, scales, 4)
    got = kpool_read_packed_cache(cache, table, 73, 4)
    torch.testing.assert_close(got[0].float(), values.float(), rtol=0, atol=0)
    torch.testing.assert_close(got[1], scales, rtol=0, atol=0)
    assert torch.equal(kpool_pooled_block_offsets(table, 4, 16), table)
    assert not cache[0].count_nonzero()


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
@pytest.mark.parametrize('heads', [32, 64])
@pytest.mark.parametrize('groups', [1, 15, 16, 17, 63, 64, 65, 513])
def test_compact_scoring_matches_fp32_reference(heads, groups):
    from lmdeploy.pytorch.backends.cuda.kpool import kpool_score_paged_cuda
    torch.manual_seed(42)
    rows, pages = 4, (groups + 15) // 16
    cache = torch.zeros((pages * 2 + 1, 16, 1, 132), dtype=torch.uint8, device='cuda')
    values, scales = kpool_packed_cache_views(cache, 128)
    values.copy_(torch.randn_like(values.float()).to(values.dtype))
    scales.copy_(torch.rand_like(scales) / 16)
    table = torch.stack([torch.randperm(pages*2, device='cuda')[:pages]+1 for _ in range(rows)]).int()
    q = torch.randn(rows, heads, 128, device='cuda').to(torch.float8_e4m3fn)
    weight = torch.randn(rows, heads, device='cuda')
    lengths = torch.tensor([groups, max(0, groups-3), 0, 1], device='cuda')
    out = kpool_score_paged_cuda(q, weight, cache, lengths, table, page_size=16)
    for row, length in enumerate(lengths.tolist()):
        k, scale = kpool_read_packed_cache(cache, table[row], length, 4)
        expected = ((q[row].float() @ k.float().T).relu() * weight[row, :, None]).sum(0) * scale[:, 0]
        torch.testing.assert_close(out[row, :length], expected, rtol=2e-5, atol=2e-5)
        assert out[row, length:].isneginf().all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
def test_raw_ring_graph_replay_changed_history():
    keys = torch.zeros(4, 8, 128, dtype=torch.bfloat16, device='cuda')
    scores = keys.clone()
    ids = torch.tensor([2, -1], device='cuda')
    history = torch.tensor([255, 0], device='cuda')
    lengths = torch.tensor([4, 0], device='cuda')
    starts = torch.tensor([0, 4], device='cuda')
    new = torch.randn(4, 128, dtype=torch.bfloat16, device='cuda')
    def step():
        write_raw_ring(keys, scores, new, new, ids, history, lengths, starts)
        return read_raw_tail(keys, scores, ids, history + lengths)
    for _ in range(3):
        step()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        result = step()
    for h in [255, 258, 263, 3, 64]:
        history[0] = h
        new.normal_()
        graph.replay()
        n = (h + 4) % 4
        torch.testing.assert_close(result[0][0, :n], new[4-n:], rtol=0, atol=0)
        assert not keys[0].count_nonzero()


@pytest.mark.parametrize('device', DEVICES)
def test_full_update_wrapper_matches_stateless_tail_oracle(device):
    from types import SimpleNamespace

    from lmdeploy.pytorch.models.glm5_next import Glm5NextSparseAttention
    torch.manual_seed(43)
    cache = torch.zeros(20, 16, 1, 132, device=device, dtype=torch.uint8)
    reference_cache = cache.clone()
    ape = torch.randn(4, 128, device=device)
    indexer = SimpleNamespace(get_block_cache=lambda: cache, project_key=lambda x: x,
                              project_compress_score=lambda x: x * .125,
                              index_kpool_compress_ape=ape, scale_fmt='ue8m0', head_dim=128)
    model = SimpleNamespace(indexer=indexer, index_kpool=4)
    ids = torch.tensor([1], device=device)
    table = torch.randperm(19, device=device)[None]+1
    ring = tuple(torch.zeros(3, 8, 128, device=device, dtype=torch.bfloat16) for _ in range(2))
    history_keys = torch.randn(600, 128, device=device, dtype=torch.bfloat16)
    # Long and short prefills, rejection/revisit, cross-owner, prefix restore.
    for h, q, decode in [(0, 258, False), (258, 4, True), (259, 4, True),
                          (260, 4, True), (263, 4, True), (264, 56, False),
                          (320, 1, False), (321, 4, True)]:
        metadata = SimpleNamespace(kv_seqlens=torch.tensor([h+q], device=device),
                                   q_seqlens=torch.tensor([q], device=device),
                                   cu_seqlens_q=torch.tensor([0, q], device=device),
                                   block_offsets=table, is_decoding=decode)
        tail = tuple(torch.zeros(2, 4, 128, device=device, dtype=torch.bfloat16) for _ in range(2))
        n = h % 4
        tail[0][1, :n] = history_keys[h-n:h]
        tail[1][1, :n] = history_keys[h-n:h] * .125
        hidden = history_keys[h:h+q][None]
        indexer.get_block_cache = lambda: reference_cache
        Glm5NextSparseAttention._update_kpool_cache(model, hidden, tail, torch.ones_like(ids), metadata)
        indexer.get_block_cache = lambda: cache
        Glm5NextSparseAttention._update_kpool_cache(model, hidden, ring, ids, metadata)
        assert torch.equal(cache, reference_cache)
        restored = read_raw_tail(*ring, ids, metadata.kv_seqlens)
        for got, expected in zip(restored, tail):
            torch.testing.assert_close(got[0], expected[1], rtol=0, atol=0)
        if h + q == 320:
            # Whole-slot prefix copy to a new runtime slot needs no metadata translation.
            for state in ring:
                state[2].copy_(state[1])
            ids.fill_(2)


@pytest.mark.parametrize('device', DEVICES)
def test_raw_ring_decode_padding_cannot_overwrite_live_scratch(device):
    from types import SimpleNamespace

    from lmdeploy.pytorch.models.glm5_next import Glm5NextSparseAttention

    torch.manual_seed(179)
    cache = torch.zeros(9, 16, 1, 132, device=device, dtype=torch.uint8)
    ape = torch.randn(4, 128, device=device)
    indexer = SimpleNamespace(get_block_cache=lambda: cache, project_key=lambda x: x,
                              project_compress_score=lambda x: x * .125,
                              index_kpool_compress_ape=ape, scale_fmt='ue8m0', head_dim=128)
    model = SimpleNamespace(indexer=indexer, index_kpool=4)
    ring = tuple(torch.zeros(3, 8, 128, device=device, dtype=torch.bfloat16) for _ in range(2))
    ids = torch.tensor([1, -1], device=device)
    table = torch.tensor([[1, 2, 3, 4], [0, 0, 0, 0]], device=device)
    hidden = torch.randn(1, 8, 128, device=device, dtype=torch.bfloat16)
    meta = SimpleNamespace(kv_seqlens=torch.tensor([4, 4], device=device),
                           q_seqlens=torch.tensor([4, 4], device=device),
                           cu_seqlens_q=torch.tensor([0, 4, 8], device=device),
                           block_offsets=table, is_decoding=True)
    Glm5NextSparseAttention._update_kpool_cache(model, hidden, ring, ids, meta)
    actual = cache.clone()
    assert not cache[0].count_nonzero()
    assert not ring[0][0].count_nonzero()
    cache.zero_()
    # Independent one-request execution from the same clean history.
    single = tuple(torch.zeros_like(x) for x in ring)
    meta.kv_seqlens = meta.kv_seqlens[:1]
    meta.q_seqlens = meta.q_seqlens[:1]
    meta.cu_seqlens_q = meta.cu_seqlens_q[:2]
    meta.block_offsets = table[:1]
    Glm5NextSparseAttention._update_kpool_cache(model, hidden[:, :4], single, ids[:1], meta)
    torch.testing.assert_close(cache, actual, rtol=0, atol=0)
    for state, expected in zip(ring, single):
        torch.testing.assert_close(state[1], expected[1], rtol=0, atol=0)


@pytest.mark.parametrize('device', DEVICES)
def test_mtp_projects_raw_tokens_once_for_pool_and_token_cache(monkeypatch, device):
    from types import SimpleNamespace

    from lmdeploy.pytorch.models import glm5_next

    torch.manual_seed(271)
    pooled = torch.zeros(5, 16, 1, 132, device=device, dtype=torch.uint8)
    raw = torch.zeros(5, 64, 2, 128, device=device, dtype=torch.bfloat16)
    calls = []

    def project_key(x):
        calls.append('key')
        return x

    def project_score(x):
        calls.append('score')
        return x * .125

    layer = glm5_next.Glm5NextMTPAttention.__new__(glm5_next.Glm5NextMTPAttention)
    torch.nn.Module.__init__(layer)
    layer.index_kpool = 4
    layer.indexer = SimpleNamespace(get_block_cache=lambda: pooled, project_key=project_key,
                                   project_compress_score=project_score,
                                   index_kpool_compress_ape=torch.zeros(4, 128, device=device),
                                   scale_fmt='ue8m0', head_dim=128)
    layer._token_cache_binding = SimpleNamespace(cache_name='tokens', consumer_row=0)
    ctx = SimpleNamespace(block_caches={'tokens': [raw]})
    monkeypatch.setattr(glm5_next, 'get_step_ctx_manager',
                        lambda: SimpleNamespace(current_context=lambda: ctx))
    hidden = torch.randn(1, 4, 128, device=device, dtype=torch.bfloat16)
    meta = SimpleNamespace(kv_seqlens=torch.tensor([4], device=device),
                           q_seqlens=torch.tensor([4], device=device),
                           cu_seqlens_q=torch.tensor([0, 4], device=device),
                           block_offsets=torch.tensor([[1, 2]], device=device), is_decoding=False)
    layer._update_kpool_cache(hidden, None, None, meta)
    assert calls == ['key', 'score']
    torch.testing.assert_close(raw[1, :4, 0], hidden[0], rtol=0, atol=0)
    torch.testing.assert_close(raw[1, :4, 1], hidden[0] * .125, rtol=0, atol=0)


@pytest.mark.parametrize('compact', [False, True])
@pytest.mark.parametrize('pybind', [False, True])
def test_deepgemm_mqa_column_capability(monkeypatch, compact, pybind):
    from types import SimpleNamespace

    from lmdeploy.pytorch.backends.cuda import kpool

    def legacy(q, kv, weights, start, end, clean_logits=True):
        pass

    def modern(q, kv, weights, start, end, clean_logits=True, max_seqlen_k=None):
        pass

    func = modern if compact else legacy
    if pybind:
        func.__doc__ = ('fp8_mqa_logits(..., max_seqlen_k: int = 0)' if compact
                        else 'fp8_mqa_logits(..., clean_logits: bool = True)')

        def no_signature(_):
            raise ValueError('pybind builtin')

        monkeypatch.setattr(kpool.inspect, 'signature', no_signature)
    monkeypatch.setattr(kpool, '_get_deep_gemm', lambda: SimpleNamespace(fp8_mqa_logits=func))
    kpool._mqa_has_local_columns.cache_clear()
    try:
        assert kpool._mqa_has_local_columns() is compact
    finally:
        kpool._mqa_has_local_columns.cache_clear()
