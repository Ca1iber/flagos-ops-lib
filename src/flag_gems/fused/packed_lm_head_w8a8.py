# SPDX-License-Identifier: Apache-2.0
"""Inference-only W8A8 head; lossy encodings, caller-owned runtime buffers.

K2048, positive M<=64 and bounded positive N. No model cache, autotune,
serving policy or implicit fallback. Input must be finite; prepare weights
before graph capture and repack after mutation.
"""
import torch
import triton
import triton.language as tl
from flag_gems.runtime import torch_device_fn
from .packed_bundle import _check_device,_check_inference,_check_writes
from .packed_swiglu import _check_tensor,_input_rows
from .packed_mlp_m64_w8a8 import _typed
from .packed_mlp_w8a8 import quantize_input

@triton.jit
def _quantize_weight(W,Q,S,BAD):
    n=tl.program_id(0);k=tl.arange(0,2048)
    w=tl.load(W+n*2048+k).to(tl.float32)
    tl.store(BAD+n,tl.sum((~(tl.abs(w)<float('inf'))).to(tl.int32),0))
    amax=tl.max(tl.abs(w),0);scale=tl.where(amax>0,tl.maximum(amax/127.,1e-30),1.)
    v=w/scale;q=tl.floor(tl.abs(v)+.5)*tl.where(v<0,-1.,1.)
    q=tl.minimum(tl.maximum(q,-127.),127.).to(tl.int8)
    offset=(n//128)*262144+(k//32)*4096+(n%128)*32+k%32
    tl.store(Q+offset,q);tl.store(S+n,scale)

@triton.jit
def _head_int8(X,XS,W,WS,Y,M:tl.constexpr,N:tl.constexpr,BM:tl.constexpr,BN:tl.constexpr,EVEN_N:tl.constexpr):
    pm=tl.program_id(0);pn=tl.program_id(1)
    rows=pm*BM+tl.arange(0,BM);cols=pn*BN+tl.arange(0,BN);rk=tl.arange(0,64)
    ap=X+rows[:,None]*64+rk[None,:]
    bp=W+(cols[None,:]//128)*262144+(rk[:,None]//32)*4096+(cols[None,:]%128)*32+rk[:,None]%32
    acc=tl.zeros((BM,BN),tl.int32)
    for _ in range(32):
        a=tl.load(ap,rows[:,None]<M,other=0)
        if EVEN_N:b=tl.load(bp)
        else:b=tl.load(bp,cols[None,:]<N,other=0)
        acc=tl.dot(a,b,acc,out_dtype=tl.int32)
        ap+=M*64;bp+=8192
    if EVEN_N:sw=tl.load(WS+cols)
    else:sw=tl.load(WS+cols,cols<N,other=0.)
    y=acc.to(tl.float32)*tl.load(XS+rows,rows<M,other=1.)[:,None]*sw[None,:]
    if EVEN_N:
        tl.store(Y+rows[:,None]*N+cols[None,:],y.to(tl.bfloat16),rows[:,None]<M)
    else:
        tl.store(Y+rows[:,None]*N+cols[None,:],y.to(tl.bfloat16),(rows[:,None]<M)&(cols[None,:]<N))


def _packed_shape(n):
    if n<=0 or triton.cdiv(n,128)*262144>=2**31:
        raise ValueError('Head N must be positive and padded weight count < 2**31')
    return (triton.cdiv(n,128),64,128,32)


def pack_lm_head_weight_int8(weight):
    """Encode finite frozen BF16 [N,2048] weights once, outside capture.

    Symmetric per-output-channel maxabs/127, half-away rounding, zero rows
    scale 1. Returns padded INT8 panels and FP32 [N] scales. Padding is zero.
    """
    _check_device(weight)
    if weight.ndim!=2:raise ValueError('Head weight must be 2D')
    n=weight.shape[0];shape=_packed_shape(n)
    _check_tensor(weight,(n,2048),'head weight',contiguous=True)
    _check_inference(weight)
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError('Head weights must be prepared before capture')
    with torch_device_fn.device(weight.device):
        packed=torch.zeros(shape,dtype=torch.int8,device=weight.device)
        scales=torch.empty((n,),dtype=torch.float32,device=weight.device)
        bad=torch.empty((n,),dtype=torch.int32,device=weight.device)
        _quantize_weight[(n,)](weight,packed,scales,bad,num_warps=4,num_stages=1)
        if bool((bad.cpu()!=0).any()):raise ValueError('Head weights must be finite')
    return packed,scales


def packed_lm_head_w8a8(x,packed_weight,weight_scales,*,input_quant_workspace,
                         input_scale_workspace,out=None):
    """Return full BF16 [M,N] logits; no token filtering or sampling changes.

    Buffers: Q[32,M,64] INT8 and scale[M] FP32, disjoint from all reads and
    each other/output. Runtime encoding is part of every call. INT32 dot
    scales in FP32 then rounds BF16. Weights are lossy and model accuracy
    requires separate validation. Graph-safe after warmup; only optional
    out=None allocates an output tensor.
    """
    _check_device(x);m=_input_rows(x,2048,contiguous=True)
    if m>64:raise ValueError('W8A8 head supports 1 <= M <= 64')
    if weight_scales.ndim!=1:raise ValueError('Head scales must be 1D')
    n=weight_scales.numel();shape=_packed_shape(n)
    _typed(packed_weight,shape,torch.int8,'head INT8 weight',x.device)
    _typed(weight_scales,(n,),torch.float32,'head weight scales',x.device)
    _typed(input_quant_workspace,(32,m,64),torch.int8,'head input quant workspace',x.device)
    _typed(input_scale_workspace,(m,),torch.float32,'head input scales',x.device)
    writes=(('head input quant workspace',input_quant_workspace),('head input scales',input_scale_workspace))
    if out is not None:
        _check_tensor(out,(m,n),'head out',contiguous=True,device=x.device);writes+=(('head out',out),)
    _check_inference(x,packed_weight,weight_scales,input_quant_workspace,input_scale_workspace,out)
    _check_writes((('x',x),('head INT8 weight',packed_weight),('head scales',weight_scales)),writes)
    bm,bn,warps,stages=(32,64,2,5) if m<=32 else (64,128,4,4)
    with torch_device_fn.device(x.device):
        if out is None:out=torch.empty((m,n),dtype=x.dtype,device=x.device)
        quantize_input[(m,)](x,input_quant_workspace,input_scale_workspace,m,num_warps=4,num_stages=1)
        _head_int8[(triton.cdiv(m,bm),triton.cdiv(n,bn))](
            input_quant_workspace,input_scale_workspace,packed_weight,weight_scales,out,m,n,
            BM=bm,BN=bn,EVEN_N=(n%128==0),num_warps=warps,num_stages=stages)
    return out


__all__=['pack_lm_head_weight_int8','packed_lm_head_w8a8']
