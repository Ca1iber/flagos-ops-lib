# SPDX-License-Identifier: Apache-2.0
"""Inference-only MetaX W8A8 gate/up + BF16 down for positive M<=2048.

Lossy input/weight encodings; caller supplies buffers and frozen packed
weights. No implicit model cache, autotune, fallback, or serving policy.
Input must be finite. Use pack_gate_up_weight_int8 for finite frozen weights.
"""
import torch
import triton
import triton.language as tl
from flag_gems.runtime import torch_device_fn
from .packed_bundle import _check_device,_check_inference,_check_writes,_packed_nt_kernel
from .packed_swiglu import _check_tensor,_input_rows
from .packed_mlp_m64_w8a8 import _typed,pack_gate_up_weight_int8

@triton.jit(do_not_specialize=["M"])
def quantize_input(X,Q,S,M):
    row=tl.program_id(0);k=tl.arange(0,2048)
    x=tl.load(X+row*2048+k).to(tl.float32)
    amax=tl.max(tl.abs(x),0)
    scale=tl.where(amax>0,tl.maximum(amax/127.,1e-30),1.)
    u=x/scale
    value=tl.floor(tl.abs(u)+.5)*tl.where(u<0,-1.,1.)
    q=tl.minimum(tl.maximum(value,-127.),127.).to(tl.int8)
    tl.store(Q+(k//64)*M*64+row*64+k%64,q)
    tl.store(S+row,scale)

@triton.jit(do_not_specialize=["M"])
def gate_int8(X,XS,W,WS,Z,M,EVEN_M:tl.constexpr):
    pid=tl.program_id(0)
    gm=tl.cdiv(M,128);gn=96;group=pid//(8*gn)
    size=tl.minimum(gm-group*8,8)
    pm=group*8+pid%size;pn=(pid%(8*gn))//size
    rows=pm*128+tl.arange(0,128)
    cols=tl.arange(0,128);rk=tl.arange(0,64)
    ap=X+rows[:,None]*64+rk[None,:]
    bp=W+pn*262144+(rk[:,None]//32)*4096+cols[None,:]*32+rk[:,None]%32
    acc=tl.zeros((128,128),tl.int32)
    for _ in range(32):
        if EVEN_M:
            a=tl.load(ap)
        else:
            a=tl.load(ap,mask=rows[:,None]<M,other=0)
        acc=tl.dot(a,tl.load(bp),acc,out_dtype=tl.int32)
        ap+=M*64;bp+=8192
    scaled=acc.to(tl.float32)*tl.load(XS+rows,mask=rows<M,other=1.)[:,None]*tl.load(WS+pn*128+cols)[None,:]
    rounded=scaled.to(tl.bfloat16).to(tl.float32)
    gate,up=tl.split(tl.reshape(rounded,(128,64,2)))
    h=pn*64+tl.arange(0,64)
    dst=pn*M*64+rows[:,None]*64+h[None,:]%64
    if EVEN_M:
        tl.store(Z+dst,(gate*tl.sigmoid(gate)*up).to(tl.bfloat16))
    else:
        tl.store(Z+dst,(gate*tl.sigmoid(gate)*up).to(tl.bfloat16),mask=rows[:,None]<M)


def packed_mlp_w8a8(x,gate_weight_int8,weight_scales,down_packed,*,
                    input_quant_workspace,input_scale_workspace,
                    activation_workspace,out=None):
    """Compute BF16 [M,2048] output, 1<=M<=2048; gate/up is quantized.

    Rowwise X encoding and K64 stores run each call. The gate projection
    uses INT32 accumulation then FP32 row/column scales, BF16 rounding,
    FP32 SiLU*up and BF16 Z. Existing BF16 down consumes packed Z.
    All supplied buffers must be disjoint from reads and one another.
    Shapes: Q[32,M,64] INT8, Xscale[M] FP32, Z[96,M,64] BF16.
    Graph-safe after warmup; no hidden buffer allocation except optional out.
    No model-level accuracy or speed guarantee is implied by this API.
    """
    _check_device(x)
    m=_input_rows(x,2048,contiguous=True)
    if m>2048:
        raise ValueError('W8A8 MLP supports 1 <= M <= 2048')
    _check_tensor(down_packed,(32,96,64,64),'down packed',contiguous=True,device=x.device)
    _typed(gate_weight_int8,(96,64,128,32),torch.int8,'gate INT8',x.device)
    _typed(weight_scales,(12288,),torch.float32,'weight scales',x.device)
    _typed(input_quant_workspace,(32,m,64),torch.int8,'input quant workspace',x.device)
    _typed(input_scale_workspace,(m,),torch.float32,'input scale workspace',x.device)
    _check_tensor(activation_workspace,(96,m,64),'activation workspace',contiguous=True,device=x.device)
    writes=(('input quant workspace',input_quant_workspace),('input scale workspace',input_scale_workspace),
            ('activation workspace',activation_workspace))
    if out is not None:
        _check_tensor(out,(m,2048),'out',contiguous=True,device=x.device)
        writes+=(('out',out),)
    _check_inference(x,gate_weight_int8,weight_scales,down_packed,input_quant_workspace,input_scale_workspace,activation_workspace,out)
    _check_writes((('x',x),('gate INT8',gate_weight_int8),('weight scales',weight_scales),('down packed',down_packed)),writes)
    with torch_device_fn.device(x.device):
        if out is None:
            out=torch.empty((m,2048),device=x.device,dtype=x.dtype)
        quantize_input[(m,)](x,input_quant_workspace,input_scale_workspace,m,num_warps=4,num_stages=1)
        gate_int8[(triton.cdiv(m,128)*96,)](
            input_quant_workspace,input_scale_workspace,gate_weight_int8,weight_scales,activation_workspace,m,
            EVEN_M=(m%128==0),num_warps=8,num_stages=2)
        _packed_nt_kernel[(triton.cdiv(m,64)*32,)](
            activation_workspace,down_packed,out,m,2048,6144,
            BM=64,BN=64,BK=64,GROUP_M=8,PACK_A=True,EVEN_M=(m%64==0),num_warps=4,num_stages=4)
    return out


__all__=['pack_gate_up_weight_int8','packed_mlp_w8a8']
