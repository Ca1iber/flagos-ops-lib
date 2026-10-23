# SPDX-License-Identifier: Apache-2.0
"""Experimental MetaX M64 W8A8 MLP. Lossy frozen weights; caller owns buffers.

Input encoding is part of each call. Native INT32 dot is scaled only after
accumulation, then existing BF16 projection/activation/down boundaries apply.
No implicit fallback, model cache, quantization CLI setting, or autotune.
"""
import torch
import triton
import triton.language as tl
from flag_gems.runtime import torch_device_fn
from .packed_bundle import _check_device,_check_inference,_check_writes
from .packed_swiglu import _check_tensor
from .packed_mlp_m64 import _m64_down_split4,_m64_down_combine

@triton.jit
def _quantize_gate_weight(X,W,S,BAD):
    n=tl.program_id(0);k=tl.arange(0,2048)
    x=tl.load(X+n*2048+k).to(tl.float32)
    good=tl.abs(x)<float('inf')
    tl.store(BAD+n,tl.sum((~good).to(tl.int32),0))
    amax=tl.max(tl.abs(x),0)
    scale=tl.where(amax>0,tl.maximum(amax/127.,1e-30),1.)
    u=x/scale
    value=tl.floor(tl.abs(u)+.5)*tl.where(u<0,-1.,1.)
    q=tl.minimum(tl.maximum(value,-127.),127.).to(tl.int8)
    inter=(n%6144)*2+n//6144
    address=(inter//128)*262144+(k//32)*4096+(inter%128)*32+k%32
    tl.store(W+address,q);tl.store(S+inter,scale)

@triton.jit
def quantize_x64_packed(X,Q,S):
    row=tl.program_id(0);k=tl.arange(0,2048)
    x=tl.load(X+row*2048+k).to(tl.float32)
    amax=tl.max(tl.abs(x),0)
    scale=tl.where(amax>0,tl.maximum(amax/127.,1e-30),1.)
    u=x/scale
    value=tl.floor(tl.abs(u)+.5)*tl.where(u<0,-1.,1.)
    q=tl.minimum(tl.maximum(value,-127.),127.).to(tl.int8)
    tl.store(Q+(k//64)*4096+row*64+k%64,q);tl.store(S+row,scale)

@triton.jit
def _m64_gate_silu_w8a8_packa(X, XSCALE, W, SCALE, Z):
    pn=tl.program_id(0)
    rm=tl.arange(0,64)
    cols=tl.arange(0,64)
    rk=tl.arange(0,64)
    ap=X+rm[:,None]*64+rk[None,:]
    bp=(W+(pn//2)*128*2048+(rk[:,None]//32)*128*32
        +((pn%2)*64+cols[None,:])*32+rk[:,None]%32)
    acc=tl.zeros((64,64),tl.int32)
    for _ in range(32):
        acc=tl.dot(tl.load(ap),tl.load(bp),acc,out_dtype=tl.int32)
        ap+=4096
        bp+=128*64
    scale=tl.load(SCALE+pn*64+cols)
    rounded=(acc.to(tl.float32)*tl.load(XSCALE+rm)[:,None]*scale[None,:]).to(tl.bfloat16).to(tl.float32)
    gate,up=tl.split(tl.reshape(rounded,(64,32,2)))
    hn=pn*32+tl.arange(0,32)
    dst=(hn[None,:]//64)*4096+rm[:,None]*64+hn[None,:]%64
    tl.store(Z+dst,(gate*tl.sigmoid(gate)*up).to(tl.bfloat16))

def _typed(tensor,shape,dtype,name,device):
    if (tuple(tensor.shape)!=shape or tensor.dtype!=dtype or tensor.device!=device
            or not tensor.is_contiguous()):
        raise ValueError(f'{name} must be contiguous {dtype} {shape} on {device}')


def pack_gate_up_weight_int8(weight):
    """Quantize finite frozen BF16 [12288,2048] weights once, before capture.

    Returns INT8 [96,64,128,32] and interleaved FP32 [12288] scales.
    Symmetric rowwise maxabs/127; zero rows have scale 1. Repack if changed.
    """
    _check_device(weight)
    _check_tensor(weight,(12288,2048),'gate/up weight',contiguous=True)
    _check_inference(weight)
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError('Quantized gate weights must be prepared before graph capture')
    with torch_device_fn.device(weight.device):
        packed=torch.empty((96,64,128,32),device=weight.device,dtype=torch.int8)
        scales=torch.empty((12288,),device=weight.device,dtype=torch.float32)
        bad=torch.empty((12288,),device=weight.device,dtype=torch.int32)
        _quantize_gate_weight[(12288,)](weight,packed,scales,bad,num_warps=4,num_stages=1)
        # One startup-only synchronization; never part of serving or capture.
        if bool((bad.cpu()!=0).any()):
            raise ValueError('W8A8 requires finite frozen gate/up weights')
    return packed,scales


def packed_mlp_m64_w8a8(x,gate_int8,weight_scales,down_packed,*,
                       input_quant_workspace,input_scale_workspace,
                       activation_workspace,partial_workspace,out=None):
    """Finite BF16 [64,2048] MLP with runtime rowwise X quantization.

    All mutable workspaces belong to caller and must outlive graph replay.
    No quantization cache or hidden allocations apart from optional output.
    """
    _check_device(x)
    for name,tensor,shape in (
        ('x',x,(64,2048)),('down packed',down_packed,(32,96,64,64)),
        ('activation workspace',activation_workspace,(96,64,64)),
    ):
        _check_tensor(tensor,shape,name,contiguous=True,device=x.device)
    _typed(gate_int8,(96,64,128,32),torch.int8,'gate INT8',x.device)
    _typed(weight_scales,(12288,),torch.float32,'weight scales',x.device)
    _typed(input_quant_workspace,(32,64,64),torch.int8,'input quant workspace',x.device)
    _typed(input_scale_workspace,(64,),torch.float32,'input scale workspace',x.device)
    _typed(partial_workspace,(4,64,2048),torch.float32,'partial workspace',x.device)
    reads=(('x',x),('gate INT8',gate_int8),('weight scales',weight_scales),('down packed',down_packed))
    writes=(('input quant workspace',input_quant_workspace),('input scale workspace',input_scale_workspace),
            ('activation workspace',activation_workspace),('partial workspace',partial_workspace))
    if out is not None:
        _check_tensor(out,(64,2048),'out',contiguous=True,device=x.device)
        writes+=(('out',out),)
    _check_inference(x,gate_int8,weight_scales,down_packed,input_quant_workspace,
                     input_scale_workspace,activation_workspace,partial_workspace,out)
    _check_writes(reads,writes)
    with torch_device_fn.device(x.device):
        if out is None:
            out=torch.empty((64,2048),device=x.device,dtype=x.dtype)
        quantize_x64_packed[(64,)](x,input_quant_workspace,input_scale_workspace,num_warps=4,num_stages=1)
        _m64_gate_silu_w8a8_packa[(192,)](input_quant_workspace,input_scale_workspace,gate_int8,
                                          weight_scales,activation_workspace,num_warps=4,num_stages=2)
        _m64_down_split4[(64,4)](activation_workspace,down_packed,partial_workspace,num_warps=4,num_stages=4)
        _m64_down_combine[(128,)](partial_workspace,out,num_warps=4)
    return out


__all__=['pack_gate_up_weight_int8','packed_mlp_m64_w8a8']
