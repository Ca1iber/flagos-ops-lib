# SPDX-License-Identifier: Apache-2.0
"""BF16-rounded full-vocabulary FP32 logits for the MetaX packed head."""
import torch
import triton
import triton.language as tl
from flag_gems.runtime import torch_device_fn
from flag_gems.fused.packed_bundle import _check_device,_check_inference,_check_writes

@triton.jit
def _packed_lm_head_logits_kernel(input_ptr,weight_ptr,output_ptr,M,N,K,stride_im,stride_ik,stride_om,stride_on,
          BLOCK_SIZE_M:tl.constexpr,BLOCK_SIZE_N:tl.constexpr,
          BLOCK_SIZE_K:tl.constexpr,PACK_N:tl.constexpr):
    pid_m=tl.program_id(0);pid_n=tl.program_id(1)
    offs_m=pid_m*BLOCK_SIZE_M+tl.arange(0,BLOCK_SIZE_M)
    offs_n=pid_n*BLOCK_SIZE_N+tl.arange(0,BLOCK_SIZE_N)
    offs_k=tl.arange(0,BLOCK_SIZE_K)
    input_ptrs=input_ptr+offs_m[:,None]*stride_im+offs_k[None,:]*stride_ik
    weight_ptrs=weight_ptr+((offs_n[None,:]//PACK_N)*(K//BLOCK_SIZE_K)*PACK_N*BLOCK_SIZE_K
                          +(offs_n[None,:]%PACK_N)*BLOCK_SIZE_K+offs_k[:,None])
    accumulator=tl.zeros((BLOCK_SIZE_M,BLOCK_SIZE_N),dtype=tl.float32)
    for k in range(0,tl.cdiv(K,BLOCK_SIZE_K)):
        input_mask_m=offs_m<M;input_mask_k=offs_k<(K-k*BLOCK_SIZE_K)
        a=tl.load(input_ptrs,mask=input_mask_m[:,None]&input_mask_k[None,:],other=0.)
        weight_mask_k=offs_k<(K-k*BLOCK_SIZE_K);weight_mask_n=offs_n<N
        b=tl.load(weight_ptrs,mask=weight_mask_k[:,None]&weight_mask_n[None,:],other=0.)
        # Same dot-then-add grouping as the installed packed LM head.
        accumulator+=tl.dot(a,b,allow_tf32=False)
        input_ptrs+=BLOCK_SIZE_K*stride_ik
        weight_ptrs+=PACK_N*BLOCK_SIZE_K
    offs_om=pid_m*BLOCK_SIZE_M+tl.arange(0,BLOCK_SIZE_M)
    offs_on=pid_n*BLOCK_SIZE_N+tl.arange(0,BLOCK_SIZE_N)
    output_ptrs=output_ptr+offs_om[:,None]*stride_om+offs_on[None,:]*stride_on
    # Preserve original BF16 boundary even if destination is FP32.
    output=accumulator.to(tl.bfloat16).to(output_ptr.dtype.element_ty)
    tl.store(output_ptrs,output,mask=(offs_m<M)[:,None]&(offs_n<N)[None,:])


def packed_lm_head_logits(x,packed_weight,*,out=None):
    """Compute FP32(BF16(X @ W.T)); generic BF16 linear/head APIs are unchanged.

    Inference-only MetaX MiniCPM head: M1/8/32/64, K2048, N130560.
    Reuses the original packed BF16 weight; no additional weight copy.
    """
    _check_device(x)
    if x.ndim!=2 or x.shape[0] not in (1,8,32,64) or x.shape[1]!=2048:
        raise ValueError("packed logits require M1/8/32/64 and K2048")
    if x.dtype!=torch.bfloat16 or not x.is_contiguous():
        raise ValueError("contiguous BF16 input required")
    w=packed_weight
    if w.shape!=(1020,64,128,32) or w.dtype!=x.dtype or w.device!=x.device or not w.is_contiguous():
        raise ValueError("original packed BF16 LM-head weight required")
    _check_inference(x,w,out)
    with torch_device_fn.device(x.device):
        if out is None:
            out=torch.empty((x.shape[0],130560),dtype=torch.float32,device=x.device)
        if out.shape!=(x.shape[0],130560) or out.dtype!=torch.float32 or out.device!=x.device or not out.is_contiguous():
            raise ValueError("contiguous same-device FP32 [M,130560] output required")
        _check_writes((("input",x),("weight",w)),(("output",out),))
        m=x.shape[0]
        bm,bn,warps,stages=(32,64,2,5) if m<=32 else (64,128,4,4)
        _packed_lm_head_logits_kernel[(triton.cdiv(m,bm),triton.cdiv(130560,bn))](
            x,w,out,m,130560,2048,x.stride(0),x.stride(1),out.stride(0),out.stride(1),
            BLOCK_SIZE_M=bm,BLOCK_SIZE_N=bn,BLOCK_SIZE_K=32,PACK_N=128,
            num_warps=warps,num_stages=stages)
    return out

__all__=["packed_lm_head_logits"]
