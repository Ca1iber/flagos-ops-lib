# SPDX-License-Identifier: Apache-2.0
"""Real-head MetaX tests; requires the contest model, not a reduced toy shape."""
import os
from pathlib import Path
import pytest
import torch
from safetensors import safe_open
from flag_gems.fused.packed_bundle import packed_lm_head,pack_linear_weight
from flag_gems.fused.packed_lm_head_logits import packed_lm_head_logits

@pytest.fixture(scope="module")
def head_weight():
    checkpoint=Path(os.getenv("FLAGOS_MODEL_PATH","/workspace/MiniCPM5-2B"))/"model-00000-of-00001.safetensors"
    if not checkpoint.exists() or not torch.cuda.is_available():
        pytest.skip("MetaX GPU and complete contest checkpoint required")
    with safe_open(str(checkpoint),framework="pt",device="cpu") as f:
        weight=f.get_tensor("lm_head.weight")
    assert weight.shape==(130560,2048) and weight.dtype==torch.bfloat16
    with torch.inference_mode():
        packed=pack_linear_weight(weight,panel_n=128,block_k=32).cuda().contiguous()
    return packed

@pytest.mark.parametrize("m",(1,8,32,64))
@pytest.mark.parametrize("scale",(0.02,1.,8.))
@pytest.mark.parametrize("preallocated",(False,True))
@torch.inference_mode()
def test_exact_bf16_boundary(head_weight,m,scale,preallocated):
    torch.manual_seed(20223701+m)
    x=torch.randn((m,2048),dtype=torch.bfloat16,device="cuda")*scale
    x=x.to(torch.bfloat16).contiguous()
    original=packed_lm_head(x,head_weight)
    assert original.dtype==torch.bfloat16
    out=torch.empty((m,130560),dtype=torch.float32,device="cuda") if preallocated else None
    actual=packed_lm_head_logits(x,head_weight,out=out)
    assert actual.dtype==torch.float32
    if out is not None:assert actual is out
    a,b=actual.cpu().contiguous(),original.float().cpu().contiguous()
    assert bool(torch.isfinite(a).all())
    assert torch.equal(a.view(torch.int32),b.view(torch.int32))

@pytest.mark.parametrize("case",("m","k","x_dtype","w_shape","out_dtype","out_stride"))
@torch.inference_mode()
def test_contract_rejections(head_weight,case):
    m=8 if case=="out_stride" else 1
    x=torch.zeros((m,2048),dtype=torch.bfloat16,device="cuda");w=head_weight;out=None
    if case=="m":x=torch.empty((65,2048),dtype=torch.bfloat16,device="cuda")
    if case=="k":x=torch.empty((1,2047),dtype=torch.bfloat16,device="cuda")
    if case=="x_dtype":x=x.float()
    if case=="w_shape":w=w[:1]
    if case=="out_dtype":out=torch.empty((m,130560),dtype=torch.bfloat16,device="cuda")
    if case=="out_stride":out=torch.empty((130560,m),dtype=torch.float32,device="cuda").t()
    with pytest.raises(ValueError):packed_lm_head_logits(x,w,out=out)

@torch.inference_mode()
def test_overlap_rejection(head_weight):
    out=torch.empty((1,130560),dtype=torch.float32,device="cuda")
    x=out.view(torch.bfloat16).reshape(-1)[:2048].reshape(1,2048)
    with pytest.raises(ValueError):packed_lm_head_logits(x,head_weight,out=out)

def test_autograd_rejection(head_weight):
    with torch.enable_grad():
        x=torch.zeros((1,2048),dtype=torch.bfloat16,device="cuda",requires_grad=True)
        with pytest.raises(NotImplementedError):packed_lm_head_logits(x,head_weight)

@torch.inference_mode()
def test_graph_fresh_input_and_canaries(head_weight):
    x=torch.zeros((64,2048),dtype=torch.bfloat16,device="cuda")
    n=64*130560;raw=torch.full((n+256,),13.,dtype=torch.float32,device="cuda")
    out=raw[128:128+n].view(64,130560)
    stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):packed_lm_head_logits(x,head_weight,out=out)
    stream.synchronize();graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph,stream=stream):packed_lm_head_logits(x,head_weight,out=out)
    for scale in (0.02,1.,8.):
        x.normal_(std=scale);before=x.cpu()
        graph.replay();torch.cuda.synchronize()
        actual=out.cpu().contiguous()
        ref=packed_lm_head(x,head_weight).float().cpu().contiguous()
        assert torch.equal(actual.view(torch.int32),ref.view(torch.int32))
        assert torch.equal(x.cpu(),before)
        assert bool((raw[:128].cpu()==13.).all()) and bool((raw[128+n:].cpu()==13.).all())
