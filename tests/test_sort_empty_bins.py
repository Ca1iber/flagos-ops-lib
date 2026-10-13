# Copyright 2026 FlagOS Contributors
"""Regression coverage for radix sweep's empty-bin fast path."""
import pytest
import torch

import flag_gems


def assert_sort_matches_reference(x, dim=-1, descending=False):
    ref_values, ref_indices = torch.sort(x, dim=dim, descending=descending, stable=True)
    with flag_gems.use_gems():
        values, indices = torch.sort(x, dim=dim, descending=descending, stable=True)
    assert values.dtype == x.dtype
    assert indices.dtype == torch.int64
    assert torch.equal(values, ref_values)
    assert torch.equal(indices, ref_indices)


@pytest.mark.sort_stable
@pytest.mark.parametrize("dtype", [
    torch.bool, torch.int16, torch.int32, torch.int64,
    torch.float16, torch.bfloat16, torch.float32,
])
@pytest.mark.parametrize("n", [2048, 2049])
@pytest.mark.parametrize("descending", [False, True])
def test_sort_empty_bins_dtype_and_tail(dtype, n, descending):
    raw = torch.randint(-4, 5, (2, n), device=flag_gems.device)
    if dtype == torch.bool:
        x = raw > 0
    elif dtype == torch.int64:
        x = (raw * (1 << 35)).to(dtype)
    else:
        x = raw.to(dtype)
    assert_sort_matches_reference(x, descending=descending)


@pytest.mark.sort_stable
@pytest.mark.parametrize("batch", [1, 8, 64])
@pytest.mark.parametrize("pattern", ["random", "ties", "minus_inf"])
def test_sort_empty_bins_sampling_vocab(batch, pattern):
    torch.manual_seed(20261013)
    if pattern == "ties":
        x = torch.randint(-4, 5, (batch, 130560), device=flag_gems.device).float()
    else:
        x = torch.randn((batch, 130560), device=flag_gems.device)
        if pattern == "minus_inf":
            x.masked_fill_(torch.rand_like(x) < 0.8, -float("inf"))
    assert_sort_matches_reference(x)


@pytest.mark.sort_stable
@pytest.mark.parametrize("descending", [False, True])
def test_sort_empty_bins_non_last_dim(descending):
    x = torch.randint(-4, 5, (2049, 3), device=flag_gems.device).float()
    assert_sort_matches_reference(x, dim=0, descending=descending)


@pytest.mark.sort_stable
@pytest.mark.parametrize("descending", [False, True])
def test_sort_empty_bins_singleton(descending):
    x = torch.ones((3, 1), device=flag_gems.device)
    assert_sort_matches_reference(x, descending=descending)


@pytest.mark.sort_stable
@pytest.mark.skipif(flag_gems.device != "cuda", reason="Requires CUDA-alike graph API")
def test_sort_empty_bins_graph_replay_with_changed_data():
    x = torch.randn((8, 130560), device=flag_gems.device)
    warmup_stream = torch.cuda.Stream()
    warmup_stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(warmup_stream):
        for _ in range(3):
            flag_gems.sort(x)
    torch.cuda.current_stream().wait_stream(warmup_stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        values, indices = flag_gems.sort(x)
    for pattern in ["random", "ties", "all_minus_inf"]:
        if pattern == "random":
            x.normal_()
        elif pattern == "ties":
            x.copy_(torch.randint(-4, 5, x.shape, device=x.device).float())
        else:
            x.fill_(-float("inf"))
        graph.replay()
        torch.cuda.synchronize()
        ref_values, ref_indices = torch.sort(x, stable=True)
        assert indices.dtype == torch.int64
        assert torch.equal(values, ref_values)
        assert torch.equal(indices, ref_indices)
