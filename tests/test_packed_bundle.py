# SPDX-License-Identifier: Apache-2.0
"""CPU/static checks without importing FlagGems, Triton, or device discovery.

Run directly with:
    OMP_NUM_THREADS=1 python tests/test_packed_bundle.py -q

The regular repository pytest conftest imports FlagGems. This standalone
unittest runner avoids it. CPU tensors exercise the real packing/span helpers;
metadata stubs exercise host validation/launch arguments, never GPU kernels.
"""

import ast
import itertools
import math
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
import unittest

try:
    import torch
except ImportError:
    torch = None

ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT / "src/flag_gems/fused/packed_bundle.py"
SWIGLU = ROOT / "src/flag_gems/fused/packed_swiglu.py"
CALLBACKS = ROOT / "src/flag_gems/runtime/prepared_callbacks.py"


def _tree(path):
    return ast.parse(path.read_text(), filename=str(path))


def _function(tree, name):
    return next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name)


def _expr(source):
    return ast.dump(ast.parse(source, mode="eval").body)


def _execute_host_nodes(path, namespace, names=None):
    # Imports and decorated Triton functions are deliberately excluded.
    nodes = [
        node for node in _tree(path).body
        if isinstance(node, ast.Assign) or (
            isinstance(node, ast.FunctionDef) and not node.decorator_list
            and (names is None or node.name in names)
        )
    ]
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)


class _TensorMetadata:
    """A CUDA-shaped descriptor with no allocation, tensor data, or GPU calls."""

    pointers = itertools.count(1 << 32, 1 << 31)

    def __init__(self, shape, *, dtype=None, device=None, pointer=None,
                 contiguous=True, requires_grad=False):
        self.shape = tuple(shape)
        self.ndim = len(self.shape)
        self.dtype = torch.bfloat16 if dtype is None else dtype
        self.device = torch.device("cuda:0") if device is None else device
        self.is_cuda = self.device.type == "cuda"
        self.requires_grad = requires_grad
        self.pointer = next(self.pointers) if pointer is None else pointer
        self.contiguous = contiguous

    def is_contiguous(self):
        return self.contiguous

    def stride(self, dim=None):
        strides = tuple(math.prod(self.shape[index + 1:]) for index in range(self.ndim))
        return strides if dim is None else strides[dim]

    def data_ptr(self):
        return self.pointer

    def element_size(self):
        return 2 if self.dtype == torch.bfloat16 else 4


class _KernelRecorder:
    def __init__(self, events, name):
        self.events, self.name = events, name

    def __getitem__(self, grid):
        def launch(*args, **kwargs):
            self.events.append((self.name, grid, args, kwargs))
        return launch


def _host_namespace(events):
    namespace = {
        "torch": torch,
        "nullcontext": nullcontext,
        "device": SimpleNamespace(vendor_name="metax"),
        "torch_device_fn": SimpleNamespace(device=lambda _: nullcontext()),
        "triton": SimpleNamespace(cdiv=lambda a, b: (a + b - 1) // b),
    }
    _execute_host_nodes(SWIGLU, namespace, {"_check_tensor", "_overlaps", "pack_gate_up_input"})
    _execute_host_nodes(BUNDLE, namespace)
    for name in ("_packed_nt_kernel", "_packed_gate_up_silu_kernel", "_packed_lm_head_kernel"):
        namespace[name] = _KernelRecorder(events, name)
    return namespace


@unittest.skipIf(torch is None, "CPU torch is unavailable; static tests still run")
class PackedBundleHostTests(unittest.TestCase):
    def setUp(self):
        self.events = []
        self.host = _host_namespace(self.events)

    def test_weight_layout_independent_copy_and_detach(self):
        for n, k, panel, block in ((12, 16, 4, 8), (1, 16, 1, 16), (8, 1, 8, 1)):
            with self.subTest(shape=(n, k), panel=panel, block=block):
                weight = torch.arange(n * k, device="cpu").reshape(n, k).to(torch.bfloat16)
                packed = self.host["pack_linear_weight"](weight, panel_n=panel, block_k=block)
                self.assertEqual(tuple(packed.shape), (n // panel, k // block, panel, block))
                self.assertTrue(packed.is_contiguous())
                self.assertFalse(packed.requires_grad)
                self.assertIsNone(packed.grad_fn)
                self.assertNotEqual(packed.data_ptr(), weight.data_ptr())
                restored = packed.permute(0, 2, 1, 3).reshape(n, k)
                self.assertTrue(torch.equal(restored, weight))
                expected = packed.clone()
                weight.fill_(-1)
                self.assertTrue(torch.equal(packed, expected))

    def test_packing_contract_and_gradient_guards(self):
        pack = self.host["pack_linear_weight"]
        weight = torch.empty((8, 16), device="cpu", dtype=torch.bfloat16)
        for panel, block, error in (
            (True, 8, TypeError), (4, 1.5, TypeError),
            (0, 8, ValueError), (4, -1, ValueError),
            (3, 8, ValueError), (4, 3, ValueError),
        ):
            with self.subTest(panel=panel, block=block):
                with self.assertRaises(error):
                    pack(weight, panel_n=panel, block_k=block)
        for bad, error in (
            (weight.to(torch.float32), TypeError),
            (weight.t(), ValueError),
            (weight.reshape(-1), ValueError),
            (torch.empty((0, 16), device="cpu", dtype=torch.bfloat16), ValueError),
        ):
            with self.subTest(shape=bad.shape, dtype=bad.dtype):
                with self.assertRaises(error):
                    pack(bad, panel_n=4, block_k=8)
        weight.requires_grad_(True)
        with torch.enable_grad(), self.assertRaisesRegex(NotImplementedError, "inference"):
            pack(weight, panel_n=4, block_k=8)
        with torch.no_grad():
            result = pack(weight, panel_n=4, block_k=8)
        self.assertFalse(result.requires_grad)

    def test_byte_span_aliases_and_input_workspace_copy(self):
        storage = torch.arange(48, device="cpu").to(torch.bfloat16)
        overlaps = self.host["_overlaps"]
        self.assertTrue(overlaps(storage[4:20], storage[16:32]))
        self.assertFalse(overlaps(storage[:16], storage[16:32]))
        # The helper is conservative for strided views; public buffers are contiguous.
        self.assertTrue(overlaps(storage[::2], storage[1::2]))
        x = torch.zeros((2048, 2048), device="cpu", dtype=torch.bfloat16)
        x[0, 31], x[2047, 2047] = 3, 7
        workspace = torch.empty((64, 2048, 32), device="cpu", dtype=x.dtype)
        packed = self.host["pack_gate_up_input"](x, out=workspace)
        self.assertIs(packed, workspace)
        self.assertTrue(torch.equal(packed.permute(1, 0, 2).reshape_as(x), x))
        with self.assertRaisesRegex(ValueError, "overlap"):
            self.host["pack_gate_up_input"](x, out=x.reshape(64, 2048, 32))

    def _projection(self, n=2048):
        return {
            "x": _TensorMetadata((2048, 2048)),
            "packed_weight": _TensorMetadata((n // 128, 64, 128, 32)),
            "out": _TensorMetadata((2048, n)),
        }

    def _mlp(self):
        return {
            "x": _TensorMetadata((2048, 2048)),
            "gate_packed": _TensorMetadata((96, 64, 128, 32)),
            "down_packed": _TensorMetadata((32, 96, 64, 64)),
            "input_workspace": _TensorMetadata((64, 2048, 32)),
            "activation_workspace": _TensorMetadata((96, 2048, 64)),
            "out": _TensorMetadata((2048, 2048)),
        }

    def _lm_head(self, m=8):
        return {
            "x": _TensorMetadata((m, 2048)),
            "packed_weight": _TensorMetadata((1020, 64, 128, 32)),
            "out": _TensorMetadata((m, 130560)),
        }

    def test_every_write_read_and_write_write_alias_rejected_before_launch(self):
        for api, factory, reads, writes in (
            ("packed_projection", self._projection, ("x", "packed_weight"), ("out",)),
            ("packed_lm_head", self._lm_head, ("x", "packed_weight"), ("out",)),
            ("packed_mlp", self._mlp, ("x", "gate_packed", "down_packed"),
             ("input_workspace", "activation_workspace", "out")),
        ):
            for index, write in enumerate(writes):
                for other in reads + writes[:index]:
                    with self.subTest(api=api, write=write, other=other):
                        arguments = factory()
                        arguments[write].pointer = arguments[other].pointer + 2
                        with self.assertRaisesRegex(ValueError, "overlap"):
                            self.host[api](**arguments)
                        self.assertEqual(self.events, [])

    def test_dtype_contiguity_device_shape_and_gradient_guards_for_every_argument(self):
        for api, factory in (
            ("packed_projection", self._projection),
            ("packed_mlp", self._mlp),
            ("packed_lm_head", self._lm_head),
        ):
            for name in factory():
                for defect in ("dtype", "contiguous", "device", "shape", "grad"):
                    with self.subTest(api=api, argument=name, defect=defect):
                        arguments = factory()
                        tensor = arguments[name]
                        error = ValueError
                        if defect == "dtype":
                            tensor.dtype, error = torch.float32, TypeError
                        elif defect == "contiguous":
                            tensor.contiguous = False
                        elif defect == "device":
                            tensor.device = torch.device("cuda:1")
                        elif defect == "shape":
                            tensor.shape = tensor.shape[:-1] + (tensor.shape[-1] + 1,)
                        else:
                            tensor.requires_grad, error = True, NotImplementedError
                        with torch.enable_grad(), self.assertRaises(error):
                            self.host[api](**arguments)
                        self.assertEqual(self.events, [])

    def test_cpu_compute_and_unsupported_m_n_rejected(self):
        x = torch.empty((1, 1), device="cpu", dtype=torch.bfloat16)
        for api, arguments in (
            ("packed_projection", {"x": x, "packed_weight": x}),
            ("packed_mlp", dict(x=x, gate_packed=x, down_packed=x,
                               input_workspace=x, activation_workspace=x)),
            ("packed_lm_head", {"x": x, "packed_weight": x}),
        ):
            with self.subTest(api=api), self.assertRaisesRegex(NotImplementedError, "MetaX"):
                self.host[api](**arguments)
        for m in (0, 2, 16, 33, 128, 2048):
            with self.subTest(m=m), self.assertRaises(ValueError):
                self.host["packed_lm_head"](**self._lm_head(m))
        arguments = self._projection()
        arguments["packed_weight"] = _TensorMetadata((24, 64, 128, 32))
        with self.assertRaises(ValueError):
            self.host["packed_projection"](**arguments)
        self.assertEqual(self.events, [])

    def test_projection_launch_contract_and_output_identity(self):
        for n in (2048, 2560):
            arguments = self._projection(n)
            result = self.host["packed_projection"](**arguments)
            self.assertIs(result, arguments["out"])
            _, grid, _, config = self.events.pop()
            self.assertEqual(grid, (16 * (n // 128),))
            self.assertEqual(config, dict(BM=128, BN=128, BK=32, GROUP_M=8,
                                          PACK_A=False, num_warps=8, num_stages=2))

    def test_mlp_copy_then_two_compute_kernels(self):
        def copy_input(x, *, out):
            self.events.append(("copy", x, out))
            return out
        self.host["pack_gate_up_input"] = copy_input
        arguments = self._mlp()
        result = self.host["packed_mlp"](**arguments)
        self.assertIs(result, arguments["out"])
        self.assertEqual([event[0] for event in self.events],
                         ["copy", "_packed_gate_up_silu_kernel", "_packed_nt_kernel"])
        _, gate_grid, gate_args, gate_config = self.events[1]
        _, down_grid, down_args, down_config = self.events[2]
        self.assertEqual(gate_grid, (1536,))
        self.assertEqual(down_grid, (1024,))
        self.assertIs(gate_args[2], arguments["activation_workspace"])
        self.assertIs(down_args[0], arguments["activation_workspace"])
        self.assertEqual(gate_config, dict(BM=128, BN=64, BK=32, GROUP_M=8,
                                           num_warps=8, num_stages=2))
        self.assertEqual(down_config, dict(BM=64, BN=64, BK=64, GROUP_M=8,
                                           PACK_A=True, num_warps=4, num_stages=4))

    def test_optional_output_allocation_is_host_metadata_only(self):
        allocations = []
        def allocate(shape, *, dtype, device):
            allocations.append((shape, dtype, device))
            return _TensorMetadata(shape, dtype=dtype, device=device)
        self.host["torch"] = SimpleNamespace(
            empty=allocate, bfloat16=torch.bfloat16,
            is_grad_enabled=torch.is_grad_enabled,
        )
        self.host["pack_gate_up_input"] = lambda x, *, out: out
        for api, factory, shape in (
            ("packed_projection", self._projection, (2048, 2048)),
            ("packed_mlp", self._mlp, (2048, 2048)),
            ("packed_lm_head", self._lm_head, (8, 130560)),
        ):
            arguments = factory()
            arguments.pop("out")
            result = self.host[api](**arguments)
            self.assertEqual(result.shape, shape)
            self.assertEqual(allocations.pop(), (shape, torch.bfloat16, arguments["x"].device))

    def test_lm_head_full_vocabulary_grid_and_locked_configs(self):
        for m in (1, 8, 32, 64):
            arguments = self._lm_head(m)
            result = self.host["packed_lm_head"](**arguments)
            self.assertIs(result, arguments["out"])
            _, grid, args, config = self.events.pop()
            bm, bn, warps, stages = (32, 64, 2, 5) if m <= 32 else (64, 128, 4, 4)
            self.assertEqual(grid, (1, 130560 // bn))
            self.assertEqual(args[3:6], (m, 130560, 2048))
            self.assertEqual(config, dict(BLOCK_SIZE_M=bm, BLOCK_SIZE_N=bn,
                                          BLOCK_SIZE_K=32, PACK_N=128,
                                          num_warps=warps, num_stages=stages))


class PackedBundleStaticTests(unittest.TestCase):
    def test_accumulation_forms_and_bf16_activation_boundary(self):
        tree = _tree(BUNDLE)
        for name in ("_packed_nt_kernel", "_packed_gate_up_silu_kernel"):
            kernel = _function(tree, name)
            dots = [node for node in ast.walk(kernel) if isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Attribute) and node.func.attr == "dot"]
            self.assertEqual(len(dots), 1)
            self.assertEqual([ast.unparse(arg) for arg in dots[0].args], ["a", "b", "acc"])
            self.assertEqual(ast.dump(dots[0]), _expr(
                "tl.dot(a, b, acc, out_dtype=tl.float32, allow_tf32=False)"))
        gate = _function(tree, "_packed_gate_up_silu_kernel")
        assignments = {
            node.targets[0].id: ast.dump(node.value) for node in gate.body
            if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name)
        }
        self.assertEqual(assignments["rounded"], _expr("acc.to(tl.bfloat16).to(tl.float32)"))
        self.assertEqual(assignments["result"], _expr("gate * tl.sigmoid(gate) * up"))
        self.assertEqual(assignments["offset"], _expr(
            "pn * M * BN + rm[:, None] * BN + tl.arange(0, BN)[None, :]"))
        lm = _function(tree, "_packed_lm_head_kernel")
        additions = [node for node in ast.walk(lm) if isinstance(node, ast.AugAssign)
                     and isinstance(node.target, ast.Name) and node.target.id == "accumulator"]
        self.assertEqual(len(additions), 1)
        self.assertIsInstance(additions[0].op, ast.Add)
        self.assertEqual(ast.dump(additions[0].value), _expr("tl.dot(a, b, allow_tf32=False)"))

    def test_lm_head_physical_panel_and_original_masks(self):
        lm = _function(_tree(BUNDLE), "_packed_lm_head_kernel")
        values = {
            node.targets[0].id: ast.dump(node.value) for node in ast.walk(lm)
            if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name)
        }
        self.assertEqual(values["weight_ptrs"], _expr(
            "weight_ptr + ((offs_n[None, :] // PACK_N) * (K // BLOCK_SIZE_K) * PACK_N * BLOCK_SIZE_K"
            " + (offs_n[None, :] % PACK_N) * BLOCK_SIZE_K + offs_k[:, None])"))
        for name, expected in (
            ("input_mask_m", "offs_m < M"),
            ("input_mask_k", "offs_k < (K - k * BLOCK_SIZE_K)"),
            ("weight_mask_k", "offs_k < (K - k * BLOCK_SIZE_K)"),
            ("weight_mask_n", "offs_n < N"),
            ("output_mask_m", "offs_m < M"),
            ("output_mask_n", "offs_n < N"),
        ):
            self.assertEqual(values[name], _expr(expected))

    def test_launch_host_code_has_no_synchronization_or_host_data_reads(self):
        banned = {"synchronize", "item", "tolist", "numpy", "cpu", "print", "manual_seed"}
        for path, names in (
            (BUNDLE, {"_check_device", "_check_inference", "_check_writes",
                      "packed_projection", "packed_mlp", "packed_lm_head"}),
            (SWIGLU, {"_check_tensor", "_overlaps", "pack_gate_up_input"}),
        ):
            for name in names:
                function = _function(_tree(path), name)
                for node in ast.walk(function):
                    if isinstance(node, ast.Call):
                        target = node.func
                        called = target.attr if isinstance(target, ast.Attribute) else (
                            target.id if isinstance(target, ast.Name) else None)
                        self.assertNotIn(called, banned, f"{path.name}:{name} calls {called}")

    def test_callback_registration_delegation_fallback_and_reset(self):
        # This module has only standard-library runtime imports.
        namespace = {"__name__": "prepared_callbacks_host_test"}
        exec(compile(CALLBACKS.read_bytes(), str(CALLBACKS), "exec"), namespace)
        mm, linear, set_handlers = (
            namespace["try_prepared_matmul"], namespace["try_prepared_linear"],
            namespace["set_prepared_handlers"],
        )
        self.assertIsNone(mm("a", "b"))
        self.assertIsNone(linear("x", "w"))
        calls, output = [], object()
        def matmul(a, b, *, out=None):
            calls.append((a, b, out))
            return output
        def projection(input, weight, bias=None):
            calls.append((input, weight, bias))
            return output
        set_handlers(matmul=matmul, linear=projection)
        self.assertIs(mm("a", "b", out=output), output)
        self.assertIs(linear("x", "w", "bias"), output)
        self.assertEqual(calls, [("a", "b", output), ("x", "w", "bias")])
        with self.assertRaises(TypeError):
            set_handlers(matmul=object())
        self.assertIs(mm("a", "b"), output)  # Invalid registration did not clear hooks.
        set_handlers(matmul=lambda *args, **kwargs: None)
        self.assertIsNone(mm("a", "b"))
        self.assertIsNone(linear("x", "w"))
        set_handlers()
        self.assertEqual(namespace["_prepared_handlers"], (None, None))


if __name__ == "__main__":
    if torch is not None:
        torch.set_num_threads(1)
    unittest.main()
