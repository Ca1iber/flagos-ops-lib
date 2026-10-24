# SPDX-License-Identifier: Apache-2.0
"""Independent native-IR RMSNorm rounding followed by K64 INT8 encoding."""
import triton
import triton.language as tl

@triton.jit(do_not_specialize=['M'])
def norm_quant(X,R,W,Q,S,RO,Y,M,EPS:tl.constexpr,ADD:tl.constexpr,STORE_NORM:tl.constexpr):
    row=tl.program_id(0);k=tl.arange(0,2048)
    x=tl.load(X+row*2048+k).to(tl.float32)
    if ADD:
        x=x+tl.load(R+row*2048+k).to(tl.float32)
        tl.store(RO+row*2048+k,x.to(tl.bfloat16))
    variance=tl.sum(x*x,0)/2048.
    # Match the ACTUAL compiled native path: no intermediate BF16 cast.
    normalized=x*tl.rsqrt(variance+EPS)
    y=(normalized.to(tl.float32)*tl.load(W+k).to(tl.float32)).to(tl.bfloat16)
    if STORE_NORM:tl.store(Y+row*2048+k,y)
    value=y.to(tl.float32);amax=tl.max(tl.abs(value),0)
    scale=tl.where(amax>0,tl.maximum(amax/127.,1e-30),1.)
    u=value/scale
    rounded=tl.floor(tl.abs(u)+.5)*tl.where(u<0,-1.,1.)
    q=tl.minimum(tl.maximum(rounded,-127.),127.).to(tl.int8)
    tl.store(Q+(k//64)*M*64+row*64+k%64,q)
    tl.store(S+row,scale)

import torch
from flag_gems.runtime import torch_device_fn
from .packed_bundle import _check_device,_check_inference,_check_writes,_packed_nt_kernel
from .packed_swiglu import _check_tensor,_input_rows
from .packed_mlp_m64_w8a8 import _typed,_m64_gate_silu_w8a8_packa,_m64_down_split4,_m64_down_combine
from .packed_mlp_w8a8 import gate_int8
from .packed_projection_w8a8 import _projection_int8

def _validate(x,norm_weight,packed_weight,scales,q,s,residual,residual_out,out,eps,n,down=None,z=None,partial=None):
    _check_device(x);m=_input_rows(x,2048,contiguous=True)
    if m>2048:raise ValueError('Norm W8A8 supports M1..2048')
    if not isinstance(eps,(float,int)) or not 0<float(eps)<float('inf'):raise ValueError('epsilon must be positive finite')
    _check_tensor(norm_weight,(2048,),'norm weight',contiguous=True,device=x.device)
    _typed(packed_weight,(n//128,64,128,32),torch.int8,'packed weight',x.device)
    _typed(scales,(n,),torch.float32,'weight scales',x.device)
    _typed(q,(32,m,64),torch.int8,'quant workspace',x.device)
    _typed(s,(m,),torch.float32,'scale workspace',x.device)
    _check_tensor(out,(m,2048 if down is not None else n),'out',contiguous=True,device=x.device)
    reads=(('x',x),('norm weight',norm_weight),('packed weight',packed_weight),('weight scales',scales))
    writes=(('quant workspace',q),('scale workspace',s),('out',out))
    if residual is not None:
        _check_tensor(residual,(m,2048),'residual',contiguous=True,device=x.device)
        _check_tensor(residual_out,(m,2048),'residual out',contiguous=True,device=x.device)
        reads+=(('residual',residual),);writes+=(('residual out',residual_out),)
    elif residual_out is not None:raise ValueError('residual out requires residual')
    if down is not None:
        _check_tensor(down,(32,96,64,64),'down weight',contiguous=True,device=x.device)
        _check_tensor(z,(96,m,64),'activation workspace',contiguous=True,device=x.device)
        reads+=(('down weight',down),);writes+=(('activation workspace',z),)
        if m==64:
            _typed(partial,(4,64,2048),torch.float32,'M64 partial workspace',x.device)
            writes+=(('partial workspace',partial),)
    tensors=(x,norm_weight,packed_weight,scales,q,s,residual,residual_out,out,down,z,partial)
    _check_inference(*tensors);_check_writes(reads,writes)
    return m


def packed_rms_projection_w8a8(x,norm_weight,packed_weight,weight_scales,*,
    input_quant_workspace,input_scale_workspace,out,residual=None,residual_out=None,eps=1e-6):
    """Native compiled RMSNorm + INT8 input encoding + K2048 projection.

    N=2048/2560, positive M<=2048, finite BF16 inputs. FP32 norm*weight
    rounds BF16 before the accepted rowwise encoder; optional FP32 x+residual
    is used for variance and written BF16 to disjoint residual_out. This
    matches current compiled native rounding, not eager's intermediate cast.
    Frozen INT8 weights/scales and caller-owned disjoint workspaces required.
    No model cache, allocation, autotune, policy, or fallback. Inference only.
    """
    n=weight_scales.numel()
    if n not in (2048,2560):raise ValueError('Projection requires N2048/2560')
    q,s=input_quant_workspace,input_scale_workspace
    m=_validate(x,norm_weight,packed_weight,weight_scales,q,s,residual,residual_out,out,eps,n)
    with torch_device_fn.device(x.device):
        norm_quant[(m,)](x,residual if residual is not None else x,norm_weight,q,s,
            residual_out if residual is not None else x,x,m,float(eps),residual is not None,False,
            num_warps=4,num_stages=1,enable_fp_fusion=False)
        _projection_int8[(triton.cdiv(m,128)*(n//128),)](q,s,packed_weight,weight_scales,out,m,n,
            EVEN_M=(m%128==0),num_warps=8,num_stages=2)
    return out


def packed_add_rms_mlp_w8a8(x,residual,norm_weight,gate_weight_int8,weight_scales,down_packed,*,
    input_quant_workspace,input_scale_workspace,activation_workspace,out,residual_out,
    partial_workspace=None,eps=1e-6):
    """Native compiled add+RMSNorm+encoded gate/SiLU+original BF16 down.

    M64 retains accepted gate tile and Split-K4 down/combine. Other M retain
    accepted variable-M kernels. Caller owns all outputs; inputs remain read
    only, including residual. Same rounding/finite-input contract as projection.
    """
    if residual is None:raise ValueError('MLP requires residual')
    q,s,z=input_quant_workspace,input_scale_workspace,activation_workspace
    m=_validate(x,norm_weight,gate_weight_int8,weight_scales,q,s,residual,residual_out,out,eps,12288,down_packed,z,partial_workspace)
    with torch_device_fn.device(x.device):
        norm_quant[(m,)](x,residual,norm_weight,q,s,residual_out,x,m,float(eps),True,False,
            num_warps=4,num_stages=1,enable_fp_fusion=False)
        if m==64:
            _m64_gate_silu_w8a8_packa[(192,)](q,s,gate_weight_int8,weight_scales,z,num_warps=4,num_stages=2)
            _m64_down_split4[(64,4)](z,down_packed,partial_workspace,num_warps=4,num_stages=4)
            _m64_down_combine[(128,)](partial_workspace,out,num_warps=4)
        else:
            gate_int8[(triton.cdiv(m,128)*96,)](q,s,gate_weight_int8,weight_scales,z,m,
                EVEN_M=(m%128==0),num_warps=8,num_stages=2)
            _packed_nt_kernel[(triton.cdiv(m,64)*32,)](z,down_packed,out,m,2048,6144,
                BM=64,BN=64,BK=64,GROUP_M=8,PACK_A=True,EVEN_M=(m%64==0),num_warps=4,num_stages=4)
    return out,residual_out

__all__=['packed_rms_projection_w8a8','packed_add_rms_mlp_w8a8']
