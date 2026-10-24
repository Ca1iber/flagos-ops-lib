# SPDX-License-Identifier: Apache-2.0
"""M64 Split-K down combine fused with the next residual RMSNorm.

Preserves the down BF16 rounding boundary. Uses existing frozen weights and
caller-owned scratch; no allocations, autotuning, dispatch changes or caches.
"""
import triton
import triton.language as tl
from flag_gems.runtime import torch_device_fn
from .packed_norm_quant_w8a8 import norm_quant,_validate
from .packed_mlp_m64_w8a8 import _m64_gate_silu_w8a8_packa,_m64_down_split4
from .packed_bundle import _check_writes,_check_inference
from .packed_swiglu import _check_tensor

@triton.jit
def combine_add_norm(P,R,W,N,RO):
    row=tl.program_id(0);k=tl.arange(0,2048);off=row*2048+k
    x=tl.zeros((2048,),tl.float32)
    for part in range(4):x+=tl.load(P+part*131072+off)
    down=x.to(tl.bfloat16).to(tl.float32)
    value=down+tl.load(R+off).to(tl.float32)
    tl.store(RO+off,value.to(tl.bfloat16))
    inv=tl.rsqrt(tl.sum(value*value,0)/2048.+1.e-6)
    normalized=(value*inv*tl.load(W+k).to(tl.float32)).to(tl.bfloat16)
    tl.store(N+off,normalized)


def packed_add_rms_mlp_next_norm_m64(x,residual,norm_weight,gate_weight_int8,
    weight_scales,down_packed,next_norm_weight,*,input_quant_workspace,
    input_scale_workspace,activation_workspace,partial_workspace,
    intermediate_residual_out,out,residual_out):
    """Accepted M64 MLP -> next residual/RMSNorm, finite BF16 inference only.

    Requires hidden=2048, intermediate=6144, eps=1e-6 for both norms.
    Returns next normalized input and next residual. The intermediate MLP
    output is rounded BF16 before adding its residual, even though no BF16
    MLP-output buffer is materialized. All writes must be disjoint.
    """
    if x.shape!=(64,2048):raise ValueError('Requires M64 hidden2048')
    if residual is None:raise ValueError('Residual required')
    q,s,z=input_quant_workspace,input_scale_workspace,activation_workspace
    p,r0=partial_workspace,intermediate_residual_out
    _validate(x,norm_weight,gate_weight_int8,weight_scales,q,s,residual,r0,out,
              1.e-6,12288,down_packed,z,p)
    _check_tensor(next_norm_weight,(2048,),'next norm weight',contiguous=True,device=x.device)
    _check_tensor(residual_out,(64,2048),'next residual out',contiguous=True,device=x.device)
    _check_inference(next_norm_weight,residual_out)
    _check_writes((('x',x),('residual',residual),('norm weight',norm_weight),
        ('gate weight',gate_weight_int8),('weight scales',weight_scales),
        ('down weight',down_packed),('next norm weight',next_norm_weight)),
        (('quant workspace',q),('scale workspace',s),('activation workspace',z),
         ('partial workspace',p),('intermediate residual',r0),('out',out),
         ('residual out',residual_out)))
    with torch_device_fn.device(x.device):
        norm_quant[(64,)](x,residual,norm_weight,q,s,r0,x,64,1.e-6,True,False,
                          num_warps=4,num_stages=1,enable_fp_fusion=False)
        _m64_gate_silu_w8a8_packa[(192,)](q,s,gate_weight_int8,weight_scales,z,
                                        num_warps=4,num_stages=2)
        _m64_down_split4[(64,4)](z,down_packed,p,num_warps=4,num_stages=4)
        combine_add_norm[(64,)](p,r0,next_norm_weight,out,residual_out,
                               num_warps=4,num_stages=1,enable_fp_fusion=False)
    return out,residual_out

__all__=['packed_add_rms_mlp_next_norm_m64']
