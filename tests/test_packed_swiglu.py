# SPDX-License-Identifier: Apache-2.0
"""Contract tests for the variable-row, inference-only MetaX packed SwiGLU API."""
import pytest
import torch

import flag_gems

M, K, H = 2048, 2048, 6144
DEVICE = flag_gems.device
METAX = flag_gems.vendor_name == "metax"


def test_packed_swiglu_input_layout_cpu():
    x = (torch.arange(M * K, dtype=torch.int32).reshape(M, K) % 127).to(torch.bfloat16)
    workspace = torch.empty((64, M, 32), dtype=x.dtype)
    result = flag_gems.pack_gate_up_input(x, out=workspace)
    assert result is workspace
    assert result.is_contiguous()
    torch.testing.assert_close(result.permute(1, 0, 2).reshape(M, K), x, rtol=0, atol=0)


def test_packed_swiglu_weight_layout_cpu():
    w = (torch.arange(2 * H * K, dtype=torch.int32).reshape(2 * H, K) % 127).to(torch.bfloat16)
    packed = flag_gems.pack_gate_up_weight(w)
    assert packed.shape == (96, 64, 128, 32)
    restored = packed.reshape(96, 64, 64, 2, 32).permute(3, 0, 2, 1, 4).reshape(2 * H, K)
    torch.testing.assert_close(restored, w, rtol=0, atol=0)


@pytest.mark.parametrize("shape", [(0, K), (M,), (1, 1, K), (M, K - 1)])
def test_packed_swiglu_rejects_input_shape(shape):
    with pytest.raises(ValueError, match="shape"):
        flag_gems.pack_gate_up_input(torch.empty(shape, dtype=torch.bfloat16))


def test_packed_swiglu_rejects_dtype():
    with pytest.raises(TypeError, match="bfloat16"):
        flag_gems.pack_gate_up_weight(torch.empty((1, 1), dtype=torch.float32))


def test_packed_swiglu_rejects_noncontiguous_weights():
    weight = torch.empty((K, 2 * H), dtype=torch.bfloat16).t()
    with pytest.raises(ValueError, match="contiguous"):
        flag_gems.pack_gate_up_weight(weight)


def test_packed_swiglu_rejects_overlapping_workspace():
    x = torch.empty((M, K), dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="overlap"):
        flag_gems.pack_gate_up_input(x, out=x.view(64, M, 32))


def test_packed_swiglu_rejects_cpu_compute():
    with pytest.raises(NotImplementedError, match="MetaX"):
        flag_gems.packed_gate_up_silu(
            torch.empty((1, 1), dtype=torch.bfloat16),
            torch.empty((1,), dtype=torch.bfloat16),
        )


@pytest.mark.skipif(not METAX, reason="fixed-shape kernel is currently MetaX-only")
@torch.inference_mode()
def test_packed_swiglu_device_output_and_graph():
    x = torch.zeros((M, K), device=DEVICE, dtype=torch.bfloat16)
    w = torch.zeros((2 * H, K), device=DEVICE, dtype=torch.bfloat16)
    x[:, 0] = (torch.arange(M, device=DEVICE) % 9 - 4).to(x.dtype) / 4
    x[:, 31] = 0.5
    w[:, 0] = (torch.arange(2 * H, device=DEVICE) % 9 - 4).to(w.dtype) / 16
    w[:, 31] = (torch.arange(2 * H, device=DEVICE) % 7 - 3).to(w.dtype) / 32
    packed = flag_gems.pack_gate_up_weight(w)
    workspace = torch.empty((64, M, 32), device=DEVICE, dtype=x.dtype)
    output = torch.empty((M, H), device=DEVICE, dtype=x.dtype)
    z = torch.nn.functional.linear(x.float(), w.float()).to(x.dtype).float()
    expected = (torch.nn.functional.silu(z[:, :H]) * z[:, H:]).to(x.dtype)
    for _ in range(3):
        result = flag_gems.packed_gate_up_silu(
            x, packed, input_workspace=workspace, out=output
        )
    assert result is output
    torch.testing.assert_close(result, expected)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        result = flag_gems.packed_gate_up_silu(
            x, packed, input_workspace=workspace, out=output
        )
    graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(result, expected)


@pytest.mark.skipif(not METAX, reason="fixed-shape kernel is currently MetaX-only")
def test_packed_swiglu_rejects_gradients():
    x = torch.empty((M, K), device=DEVICE, dtype=torch.bfloat16, requires_grad=True)
    w = torch.empty((96, 64, 128, 32), device=DEVICE, dtype=torch.bfloat16)
    with torch.enable_grad(), pytest.raises(NotImplementedError, match="inference"):
        flag_gems.packed_gate_up_silu(x, w)
