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


"""Regression coverage for bounded INT32 radix histogram counts."""
import importlib

import pytest
import torch

import flag_gems


def assert_stable_result(x, descending):
    # Reference stays outside FlagGems; UInt32-dependent radix internals use it.
    values_ref, indices_ref = torch.sort(x, dim=-1, stable=True, descending=descending)
    with flag_gems.use_gems():
        values, indices = torch.sort(x, dim=-1, stable=True, descending=descending)
    assert values.dtype == x.dtype
    assert indices.dtype == torch.int64
    assert torch.equal(values, values_ref)
    assert torch.equal(indices, indices_ref)


@pytest.mark.sort_stable
@pytest.mark.parametrize("shape", [(1, 8191), (2, 8193), (64, 130560)])
@pytest.mark.parametrize("pattern", ["random", "ties", "masked"])
@pytest.mark.parametrize("descending", [False, True])
def test_sort_count32_sampling_shapes(shape, pattern, descending):
    if pattern == "ties":
        x = torch.randint(-8, 8, shape, device=flag_gems.device).float()
    else:
        x = torch.randn(shape, device=flag_gems.device)
        if pattern == "masked":
            x[:, ::4] = -float("inf")
    assert_stable_result(x, descending)


@pytest.mark.sort_stable
@pytest.mark.parametrize(
    "dtype",
    [torch.float16, torch.bfloat16, torch.float64, torch.int32, torch.int64, torch.uint8],
)
@pytest.mark.parametrize("descending", [False, True])
def test_sort_count32_other_dtypes(dtype, descending):
    x = torch.randint(-8, 8, (2, 8193), device=flag_gems.device).to(dtype)
    assert_stable_result(x, descending)


@pytest.mark.sort_stable
@pytest.mark.parametrize("shape", [(1, 8192), (2, 8193), (64, 130560)])
@pytest.mark.parametrize("descending", [False, True])
def test_sort_count32_zero_histogram(shape, descending):
    sort_module = importlib.import_module("flag_gems.ops.sort")
    m, n = shape
    x = torch.zeros(shape, dtype=torch.float32, device=flag_gems.device)
    actual = torch.zeros((m, 8, 16), dtype=torch.int32, device=flag_gems.device)
    expected = torch.zeros_like(actual)
    # +0 has ascending key 0x80000000; descending key is 0x7fffffff.
    expected[:, :7, 15 if descending else 0] = n
    expected[:, 7, 7 if descending else 8] = n
    sort_module.compute_global_hist_kernel[(m * ((n + 8191) // 8192), 1, 1)](
        x, actual, 8, m, n, 8, 1024, 16, 4, descending
    )
    assert torch.equal(actual, expected)
