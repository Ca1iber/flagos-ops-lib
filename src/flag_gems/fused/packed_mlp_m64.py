# SPDX-License-Identifier: Apache-2.0
"""M64 BF16 MLP with direct input loads and FP32 Split-K4 down projection.

Uses existing packed gate/up and down weights. All mutable workspaces are
caller-owned; warm before CUDA graph capture. Inference only, MetaX only.
"""
import torch
import triton
import triton.language as tl
from flag_gems.runtime import torch_device_fn
from .packed_bundle import _check_device, _check_inference, _check_writes
from .packed_swiglu import _check_tensor

@triton.jit
def _m64_gate_silu(X, W, Z):
    pn=tl.program_id(0)
    rm=tl.arange(0,64)
    cols=tl.arange(0,64)
    rk=tl.arange(0,64)
    ap=X+rm[:,None]*2048+rk[None,:]
    bp=(W+(pn//2)*128*2048+(rk[:,None]//32)*128*32
        +((pn%2)*64+cols[None,:])*32+rk[:,None]%32)
    acc=tl.zeros((64,64),tl.float32)
    for _ in range(32):
        acc=tl.dot(tl.load(ap),tl.load(bp),acc,out_dtype=tl.float32,allow_tf32=False)
        ap+=64
        bp+=128*64
    rounded=acc.to(tl.bfloat16).to(tl.float32)
    gate,up=tl.split(tl.reshape(rounded,(64,32,2)))
    hn=pn*32+tl.arange(0,32)
    dst=(hn[None,:]//64)*4096+rm[:,None]*64+hn[None,:]%64
    tl.store(Z+dst,(gate*tl.sigmoid(gate)*up).to(tl.bfloat16))

@triton.jit
def _m64_down_split4(Z,W,P):
    pid=tl.program_id(0)
    part=tl.program_id(1)
    rows=(pid%2)*32+tl.arange(0,32)
    rows=tl.max_contiguous(tl.multiple_of(rows,32),32)
    cols=tl.arange(0,64)
    kk=tl.arange(0,64)
    start=part*24
    ap=Z+start*4096+rows[:,None]*64+kk[None,:]
    bp=W+(pid//2)*96*4096+start*4096+cols[None,:]*64+kk[:,None]
    acc=tl.zeros((32,64),tl.float32)
    for _ in range(24):
        acc=tl.dot(tl.load(ap),tl.load(bp),acc,out_dtype=tl.float32,allow_tf32=False)
        ap+=4096
        bp+=4096
    tl.store(P+part*131072+rows[:,None]*2048+(pid//2)*64+cols[None,:],acc)

@triton.jit
def _m64_down_combine(P,Y):
    offsets=tl.program_id(0)*1024+tl.arange(0,1024)
    acc=tl.zeros((1024,),tl.float32)
    for part in range(4):
        acc+=tl.load(P+part*131072+offsets)
    tl.store(Y+offsets,acc.to(tl.bfloat16))

def packed_mlp_m64(x,gate_packed,down_packed,*,activation_workspace,
                   partial_workspace,out=None):
    """BF16 [64,2048] MLP, using packed BF16 Z and FP32 [4,64,2048] partials."""
    _check_device(x)
    for name,tensor,shape in (
        ("x",x,(64,2048)),
        ("gate packed weight",gate_packed,(96,64,128,32)),
        ("down packed weight",down_packed,(32,96,64,64)),
        ("activation workspace",activation_workspace,(96,64,64)),
    ):
        _check_tensor(tensor,shape,name,contiguous=True,device=x.device)
    p=partial_workspace
    if (p.shape!=(4,64,2048) or p.dtype!=torch.float32
            or p.device!=x.device or not p.is_contiguous()):
        raise ValueError("partial workspace must be contiguous device FP32 [4,64,2048]")
    writes=(("activation workspace",activation_workspace),("partial workspace",p))
    if out is not None:
        _check_tensor(out,(64,2048),"out",contiguous=True,device=x.device)
        writes+=(("out",out),)
    _check_inference(x,gate_packed,down_packed,activation_workspace,p,out)
    _check_writes((("x",x),("gate packed weight",gate_packed),
                   ("down packed weight",down_packed)),writes)
    with torch_device_fn.device(x.device):
        if out is None:
            out=torch.empty((64,2048),device=x.device,dtype=x.dtype)
        _m64_gate_silu[(192,)](
            x,gate_packed,activation_workspace,num_warps=4,num_stages=2)
        _m64_down_split4[(64,4)](
            activation_workspace,down_packed,p,num_warps=4,num_stages=4)
        _m64_down_combine[(128,)](p,out,num_warps=4)
    return out

__all__=["packed_mlp_m64"]
