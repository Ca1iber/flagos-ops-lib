# SPDX-License-Identifier: Apache-2.0
"""Exact bits, bounds and graph freshness for the optional tiled input producer."""
import importlib

import pytest
import torch

import flag_gems

S = importlib.import_module("flag_gems.fused.packed_swiglu")
pytestmark = pytest.mark.skipif(flag_gems.vendor_name != "metax", reason="MetaX producer")


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("m,stride,offset,poff", [
    (80, 2048, 0, 1),
    (1791, 2048, 1, 0),
    (1792, 2112, 0, 128),
    (1856, 2048, 128, 0),
    (2047, 2048, 1, 1),
    (2048, 0, 0, 128),
])
def test_tiled_input_bits_bounds_and_replay(monkeypatch, enabled, m, stride, offset, poff):
    monkeypatch.setattr(S, "_TILED_INPUT_PACK_ENABLED", enabled)
    with torch.inference_mode():
        count = offset + (m - 1) * stride + 2048 + 17
        words = torch.arange(count, device="cuda", dtype=torch.int32).to(torch.int16)
        before = words.clone()
        x = torch.as_strided(words.view(torch.bfloat16), (m, 2048), (stride, 1), offset)
        target = torch.full((poff + m * 2048 + 17,), -1234, device="cuda", dtype=torch.int16)
        out = target[poff:poff + m * 2048].view(torch.bfloat16).reshape(64, m, 32)
        assert S.pack_gate_up_input(x, out=out) is out
        torch.cuda.synchronize()
        assert torch.equal(out.view(torch.int16), x.reshape(m, 64, 32).permute(1, 0, 2).view(torch.int16))
        assert torch.equal(words, before)
        assert bool((target[:poff] == -1234).all())
        assert bool((target[poff + m * 2048:] == -1234).all())
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            S.pack_gate_up_input(x, out=out)
        words.copy_(words.roll(1))
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(out.view(torch.int16), x.reshape(m, 64, 32).permute(1, 0, 2).view(torch.int16))
        assert bool((target[:poff] == -1234).all())
        assert bool((target[poff + m * 2048:] == -1234).all())


def test_tiled_input_flag_preserves_autograd_copy(monkeypatch):
    monkeypatch.setattr(S, "_TILED_INPUT_PACK_ENABLED", True)
    with torch.enable_grad():
        x = torch.ones((1792, 2048), device="cuda", dtype=torch.bfloat16, requires_grad=True)
        out = S.pack_gate_up_input(x)
        assert out.requires_grad
        out.float().sum().backward()
        assert torch.equal(x.grad, torch.ones_like(x))
