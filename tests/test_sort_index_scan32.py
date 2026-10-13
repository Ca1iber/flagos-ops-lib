# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Bounded histogram prefixes and INT32-internal/INT64-public sort indices."""
import importlib

import pytest
import torch

import flag_gems


@pytest.mark.parametrize("bins", [2, 16, 256])
@pytest.mark.parametrize("pattern", ["empty", "first", "last", "split"])
def test_histogram_exclusive_scan32(bins, pattern):
    sort_module = importlib.import_module("flag_gems.ops.sort")
    hist = torch.zeros((2, 3, bins), dtype=torch.int32, device=flag_gems.device)
    bound = (1 << 30) - 1
    if pattern == "first":
        hist[..., 0] = bound
    elif pattern == "last":
        hist[..., -1] = bound
    elif pattern == "split":
        hist[..., 0] = bound // 2
        hist[..., -1] = bound - bound // 2
    expected = torch.cumsum(hist, dim=-1, dtype=torch.int64) - hist
    actual = torch.empty_like(hist, dtype=torch.uint32)
    sort_module._histogram_exclusive_scan_kernel[(6,)](
        hist, actual, NUM_BINS=bins, num_warps=4
    )
    with flag_gems.use_gems():
        actual64 = actual.to(torch.int64)
    assert actual.dtype == torch.uint32
    torch.testing.assert_close(actual64, expected, rtol=0, atol=0)


@pytest.mark.sort_stable
@pytest.mark.parametrize("shape", [(1, 17), (2, 17)])
@pytest.mark.parametrize("dim", [0, -1])
@pytest.mark.parametrize("dtype", [torch.bool, torch.int64])
@pytest.mark.parametrize("descending", [False, True])
def test_sort_index32_public_api(shape, dim, dtype, descending):
    raw = torch.randint(-4, 5, shape, device=flag_gems.device, dtype=torch.int64)
    x = raw.bool() if dtype == torch.bool else raw * (1 << 40)
    ref_values, ref_indices = torch.sort(x, dim=dim, stable=True, descending=descending)
    with flag_gems.use_gems():
        values, indices = torch.sort(x, dim=dim, stable=True, descending=descending)
    assert values.dtype == dtype
    assert indices.dtype == torch.int64
    assert torch.equal(values, ref_values)
    assert torch.equal(indices, ref_indices)
