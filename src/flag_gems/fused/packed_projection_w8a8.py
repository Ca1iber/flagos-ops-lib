# SPDX-License-Identifier: Apache-2.0
"""Inference-only K2048 W8A8 projections with caller-owned runtime buffers.

Positive M<=2048, N2048/2560. Lossy row/channel maxabs encodings; no model
cache, autotune, serving policy, or implicit fallback. Inputs must be finite.
"""
import torch
import triton
import triton.language as tl
from flag_gems.runtime import torch_device_fn
from .packed_bundle import _check_device, _check_inference, _check_writes
from .packed_swiglu import _check_tensor, _input_rows
from .packed_mlp_m64_w8a8 import _typed
from .packed_mlp_w8a8 import quantize_input
from .packed_lm_head_w8a8 import pack_lm_head_weight_int8 as _pack_k2048_weight

@triton.jit(do_not_specialize=['M'])
def _projection_int8(X,XS,W,WS,Y,M,N:tl.constexpr,EVEN_M:tl.constexpr):
    pid=tl.program_id(0);gm=tl.cdiv(M,128);gn=N//128;group=pid//(8*gn)
    size=tl.minimum(gm-group*8,8)
    pm=group*8+pid%size;pn=(pid%(8*gn))//size
    rows=pm*128+tl.arange(0,128);cols=tl.arange(0,128);rk=tl.arange(0,64)
    ap=X+rows[:,None]*64+rk[None,:]
    bp=W+pn*262144+(rk[:,None]//32)*4096+cols[None,:]*32+rk[:,None]%32
    acc=tl.zeros((128,128),tl.int32)
    for _ in range(32):
        if EVEN_M:a=tl.load(ap)
        else:a=tl.load(ap,rows[:,None]<M,other=0)
        acc=tl.dot(a,tl.load(bp),acc,out_dtype=tl.int32)
        ap+=M*64;bp+=8192
    y=acc.to(tl.float32)*tl.load(XS+rows,rows<M,other=1.)[:,None]*tl.load(WS+pn*128+cols)[None,:]
    tl.store(Y+rows[:,None]*N+pn*128+cols[None,:],y.to(tl.bfloat16),rows[:,None]<M)


def pack_projection_weight_int8(weight):
    """Encode frozen finite BF16 [N,2048] before capture; N2048/2560.

    Same per-output maxabs/127, half-away rounding and N128/K32 panels as
    the head codec. Returns INT8 [N/128,64,128,32] and FP32 [N] scales.
    Repack explicitly after weight mutation; source weight is not modified.
    """
    if weight.ndim != 2 or weight.shape[0] not in (2048, 2560):
        raise ValueError('Projection weight requires N=2048 or 2560')
    return _pack_k2048_weight(weight)


def packed_projection_w8a8(x, packed_weight, weight_scales, *,
                           input_quant_workspace, input_scale_workspace, out=None):
    """Compute BF16 [M,N], 1<=M<=2048, K2048, N2048/2560.

    Includes rowwise X quantization each call. INT32 accumulation scales in
    FP32 then rounds BF16. Q[32,M,64] INT8 and Xscale[M] FP32 are supplied
    by caller, disjoint from reads, output and each other. Graph-safe after
    warmup; only out=None allocates a tensor. Model accuracy and speed need
    separate validation. A smaller M is valid but not guaranteed faster.
    """
    _check_device(x)
    m = _input_rows(x, 2048, contiguous=True)
    if m > 2048:
        raise ValueError('W8A8 projection supports 1 <= M <= 2048')
    if weight_scales.ndim != 1 or weight_scales.numel() not in (2048, 2560):
        raise ValueError('Projection scales require N=2048 or 2560')
    n = weight_scales.numel()
    _typed(packed_weight, (n//128,64,128,32), torch.int8, 'projection weight', x.device)
    _typed(weight_scales, (n,), torch.float32, 'projection scales', x.device)
    _typed(input_quant_workspace, (32,m,64), torch.int8, 'input quant workspace', x.device)
    _typed(input_scale_workspace, (m,), torch.float32, 'input scale workspace', x.device)
    writes = (('input quant workspace', input_quant_workspace), ('input scale workspace', input_scale_workspace))
    if out is not None:
        _check_tensor(out, (m,n), 'projection out', contiguous=True, device=x.device)
        writes += (('projection out', out),)
    _check_inference(x, packed_weight, weight_scales, input_quant_workspace, input_scale_workspace, out)
    _check_writes((('x',x), ('projection weight',packed_weight), ('projection scales',weight_scales)), writes)
    with torch_device_fn.device(x.device):
        if out is None:
            out = torch.empty((m,n), dtype=x.dtype, device=x.device)
        quantize_input[(m,)](x, input_quant_workspace, input_scale_workspace, m, num_warps=4, num_stages=1)
        _projection_int8[(triton.cdiv(m,128)*(n//128),)](
            input_quant_workspace, input_scale_workspace, packed_weight, weight_scales, out, m, n,
            EVEN_M=(m%128==0), num_warps=8, num_stages=2)
    return out


__all__ = ['pack_projection_weight_int8', 'packed_projection_w8a8']
