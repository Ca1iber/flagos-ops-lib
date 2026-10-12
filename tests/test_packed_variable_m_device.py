# SPDX-License-Identifier: Apache-2.0
"""Standalone MetaX tests for variable rows; no vLLM service or autotuning.

Run against the checkout:
  FLAGOS_VAR_M_BASELINE_ROOT=/path/to/pre-change/flag_gems/fused \
  PYTHONPATH=src python tests/test_packed_variable_m_device.py
The explicitly supplied pre-change M2048 implementation is the regression oracle.
Do not compare with the newly installed implementation as if it were an old version.
Without a baseline fixture, these cross-version device tests are skipped.
To recreate the fixture, export packed_bundle.py and packed_swiglu.py from
FlagGems commit 3b162a7ba78b4269c62f72d05023949c1e72dbd5 using git show.
Structured data also has an independent FP32/BF16-boundary reference.
"""
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import unittest

import torch
import torch.nn.functional as F
import flag_gems as gems
from flag_gems.fused import packed_bundle, packed_swiglu

ROWS = (1, 63, 64, 65, 127, 128, 129, 511, 512, 513,
        565, 576, 1025, 1152, 1537, 1664, 1920, 2047, 2048)
MAX_M, K, H, GUARD = 2048, 2048, 6144, 128
SENTINEL = 71
RECORD = {"row_counts": ROWS, "eager": [], "graphs": [], "independent_reference": [],
          "service_integration_changed": False, "performance_measured": False}


def original_module(name):
    baseline_root = os.environ.get("FLAGOS_VAR_M_BASELINE_ROOT")
    if not baseline_root:
        raise unittest.SkipTest("Set FLAGOS_VAR_M_BASELINE_ROOT to a pre-change fixture")
    root = Path(baseline_root)
    path = root / (name + ".py")
    spec = importlib.util.spec_from_file_location("flag_gems.fused._fixed_baseline_" + name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    RECORD["baseline_" + name] = str(path)
    return module


def guarded(shape):
    count = math.prod(shape)
    allocation = torch.full((count + 2 * GUARD,), SENTINEL, device="cuda", dtype=torch.bfloat16)
    return allocation, allocation[GUARD:GUARD + count].view(shape)


def packed(weights):
    q, o, gate, down = weights
    return (gems.pack_linear_weight(q, panel_n=128, block_k=32),
            gems.pack_linear_weight(o, panel_n=128, block_k=32),
            gems.pack_gate_up_weight(gate),
            gems.pack_linear_weight(down, panel_n=64, block_k=64))


class VariableRowsDeviceTests(unittest.TestCase):
    @classmethod
    @torch.inference_mode()
    def setUpClass(cls):
        if gems.vendor_name != "metax" or not torch.cuda.is_available():
            raise unittest.SkipTest("These device tests require a MetaX instance")
        root = Path(__file__).resolve().parents[1] / "src"
        for module in (gems, packed_bundle, packed_swiglu):
            if not Path(module.__file__).resolve().is_relative_to(root):
                raise RuntimeError("Run with PYTHONPATH=src; source checkout must be tested")
        cls.old_bundle = original_module("packed_bundle")
        cls.old_swiglu = original_module("packed_swiglu")
        cls.old_xp = torch.empty((64, MAX_M, 32), device="cuda", dtype=torch.bfloat16)
        cls.old_z = torch.empty((96, MAX_M, 64), device="cuda", dtype=torch.bfloat16)
        cls.x_storage, _ = guarded((64, MAX_M, 32))
        cls.z_storage, _ = guarded((96, MAX_M, 64))
        torch.manual_seed(20261011)
        cls.dense_x = torch.randn((MAX_M, K), device="cuda", dtype=torch.bfloat16) * 0.25
        cls.dense_w = tuple(torch.randn((n, k), device="cuda", dtype=torch.bfloat16) * 0.02
                            for n, k in ((2560, K), (K, K), (2 * H, K), (K, H)))
        indices = torch.tensor([0, 31, 32, 127, 1023, 2047], device="cuda")
        cls.sparse_x = torch.zeros((MAX_M, K), device="cuda", dtype=torch.bfloat16)
        rows = torch.arange(MAX_M, device="cuda")[:, None]
        cls.sparse_x[:, indices] = ((rows + torch.arange(6, device="cuda")) % 9 - 4).to(torch.bfloat16) / 8
        weights = []
        for n in (2560, K, 2 * H):
            w = torch.zeros((n, K), device="cuda", dtype=torch.bfloat16)
            w[:, indices] = ((torch.arange(n, device="cuda")[:, None] +
                              torch.arange(6, device="cuda")) % 7 - 3).to(torch.bfloat16) / 32
            weights.append(w)
        down = torch.zeros((K, H), device="cuda", dtype=torch.bfloat16)
        r = torch.arange(K, device="cuda")
        for offset, coefficient in ((0, 0.5), (K, 0.25), (2 * K, -0.125)):
            down[r, r + offset] = coefficient
        cls.sparse_w = (*weights, down)
        RECORD["source"] = str(packed_bundle.__file__)
        props = torch.cuda.get_device_properties(0)
        RECORD["device"] = {"name": props.name, "total_memory": props.total_memory,
                            "torch": torch.__version__}

    def views(self, m):
        self.x_storage.fill_(SENTINEL)
        self.z_storage.fill_(SENTINEL)
        xp = self.x_storage[GUARD:GUARD + 64 * m * 32].view(64, m, 32)
        z = self.z_storage[GUARD:GUARD + 96 * m * 64].view(96, m, 64)
        return xp, z

    def check_guards(self, allocation, active_count):
        self.assertTrue(bool((allocation[:GUARD] == SENTINEL).all().item()))
        self.assertTrue(bool((allocation[GUARD + active_count:] == SENTINEL).all().item()))

    @torch.inference_mode()
    def baseline(self, x, weights):
        q, o, gate, down = weights
        return (self.old_bundle.packed_projection(x, q),
                self.old_bundle.packed_projection(x, o),
                self.old_swiglu.packed_gate_up_silu(x, gate, input_workspace=self.old_xp),
                self.old_bundle.packed_mlp(x, gate, down, input_workspace=self.old_xp,
                                          activation_workspace=self.old_z))

    @torch.inference_mode()
    def check_dataset(self, label, x, raw_weights, *, independent=False):
        weights = packed(raw_weights)
        expected = self.baseline(x, weights)
        if independent:
            q, o, gate, down = raw_weights
            gate_up = F.linear(x.float(), gate.float()).to(x.dtype).float()
            z = (F.silu(gate_up[:, :H]) * gate_up[:, H:]).to(x.dtype)
            refs = (F.linear(x.float(), q.float()).to(x.dtype),
                    F.linear(x.float(), o.float()).to(x.dtype), z,
                    F.linear(z.float(), down.float()).to(x.dtype))
            for kind, got, ref in zip(("qkv", "o_proj", "swiglu", "mlp"), expected, refs):
                torch.testing.assert_close(got, ref)
                RECORD["independent_reference"].append({"data": label, "kind": kind, "passed": True})
        for m in ROWS:
            xp, z = self.views(m)
            for index, (kind, n) in enumerate((("qkv", 2560), ("o_proj", K),
                                               ("swiglu", H), ("mlp", K))):
                allocation, output = guarded((m, n))
                if index < 2:
                    got = gems.packed_projection(x[:m], weights[index], out=output)
                elif index == 2:
                    got = gems.packed_gate_up_silu(x[:m], weights[2], input_workspace=xp, out=output)
                else:
                    got = gems.packed_mlp(x[:m], weights[2], weights[3],
                                         input_workspace=xp, activation_workspace=z, out=output)
                self.assertIs(got, output)
                torch.testing.assert_close(got, expected[index][:m], rtol=0, atol=0)
                self.check_guards(allocation, m * n)
                self.check_guards(self.x_storage, 64 * m * 32)
                self.check_guards(self.z_storage, 96 * m * 64)
                RECORD["eager"].append({"data": label, "kind": kind, "M": m,
                                        "bitwise_baseline": True, "guards": True})
            if m in (1, 576, 1664, 2048):
                print(f"eager data={label} M={m}: four paths bitwise-match fixed baseline; guards OK", flush=True)

    def test_01_structured_and_independent_reference(self):
        self.check_dataset("structured", self.sparse_x, self.sparse_w, independent=True)

    def test_02_dense_and_large_amplitude_regression(self):
        self.check_dataset("dense", self.dense_x, self.dense_w)
        self.check_dataset("large_amplitude", self.dense_x * 64, self.dense_w)

    @torch.inference_mode()
    def test_03_real_model_weights_regression(self):
        from safetensors import safe_open
        path = Path("/workspace/MiniCPM5-2B/model-00000-of-00001.safetensors")
        if not path.exists():
            self.skipTest("Local model checkpoint is unavailable")
        with safe_open(str(path), framework="pt", device="cpu") as f:
            prefix = "model.layers.0."
            get = lambda key: f.get_tensor(prefix + key).to(device="cuda", dtype=torch.bfloat16)
            q = torch.cat([get("self_attn." + name + "_proj.weight") for name in ("q", "k", "v")])
            weights = (q, get("self_attn.o_proj.weight"),
                       torch.cat([get("mlp.gate_proj.weight"), get("mlp.up_proj.weight")]),
                       get("mlp.down_proj.weight"))
        self.check_dataset("model_layer0", self.dense_x, weights)

    @torch.inference_mode()
    def test_04_alternating_graphs_with_shared_workspace(self):
        weights = packed(self.sparse_w)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        graphs = []
        for m in (1, 65, 565, 576, 1537, 1664, 2048):
            x = self.sparse_x[:m].clone()
            xp, z = self.views(m)
            buffers = [guarded((m, n)) for n in (2560, K, H, K)]
            def run():
                gems.packed_projection(x, weights[0], out=buffers[0][1])
                gems.packed_projection(x, weights[1], out=buffers[1][1])
                gems.packed_gate_up_silu(x, weights[2], input_workspace=xp, out=buffers[2][1])
                gems.packed_mlp(x, weights[2], weights[3], input_workspace=xp,
                                activation_workspace=z, out=buffers[3][1])
            with torch.cuda.stream(stream):
                for _ in range(2):
                    run()
            stream.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                run()
            graphs.append((m, x, graph, buffers))
        torch.cuda.synchronize()
        for factor, order in ((1, graphs), (-1, list(reversed(graphs))), (0.5, graphs)):
            padded = torch.zeros_like(self.sparse_x)
            for m, x, graph, buffers in order:
                self.x_storage.fill_(SENTINEL)
                self.z_storage.fill_(SENTINEL)
                x.copy_(self.sparse_x[:m] * factor)
                graph.replay()
                torch.cuda.synchronize()
                padded.zero_()
                padded[:m].copy_(x)
                references = self.baseline(padded, weights)
                for (allocation, output), ref in zip(buffers, references):
                    torch.testing.assert_close(output, ref[:m], rtol=0, atol=0)
                    self.check_guards(allocation, output.numel())
                self.check_guards(self.x_storage, 64 * m * 32)
                self.check_guards(self.z_storage, 96 * m * 64)
                RECORD["graphs"].append({"M": m, "factor": factor, "bitwise_baseline": True,
                                        "guards": True, "paths": ["qkv", "o_proj", "swiglu", "mlp"]})
            print(f"graph factor={factor}: alternating seven M values passed", flush=True)


if __name__ == "__main__":
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(VariableRowsDeviceTests)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    RECORD["tests_run"] = result.testsRun
    RECORD["successful"] = result.wasSuccessful()
    RECORD["failures"] = [(str(test), trace[-3000:]) for test, trace in result.failures + result.errors]
    RECORD["skipped"] = [(str(test), reason) for test, reason in result.skipped]
    destination = os.environ.get("FLAGOS_VAR_M_RESULTS")
    if destination:
        Path(destination).write_text(json.dumps(RECORD, indent=2) + "\n")
    sys.exit(0 if result.wasSuccessful() else 1)
