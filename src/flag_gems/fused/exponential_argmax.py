"""Experimental FP32 exponential-race argmax; no RNG or filtering changes."""
import torch
import triton
import triton.language as tl

@triton.jit
def _partial(P,Q,S,I,N:tl.constexpr,PS0,PS1,QS0,QS1,T:tl.constexpr,BLOCK:tl.constexpr):
    tile=tl.program_id(0)
    row=tl.program_id(1)
    col=tile*BLOCK+tl.arange(0,BLOCK)
    valid=col<N
    prob=tl.load(P+row*PS0+col*PS1,valid,other=0.)
    noise=tl.load(Q+row*QS0+col*QS1,valid,other=1.)
    score=tl.div_rn(prob,noise)
    nan=valid & (score!=score)
    first_nan=tl.min(tl.where(nan,col,2147483647),0)
    finite_score=tl.where(valid & ~nan,score,-float("inf"))
    maximum=tl.max(finite_score,0)
    first_best=tl.min(tl.where(valid & ~nan & (score==maximum),col,2147483647),0)
    chosen=tl.where(first_nan<2147483647,first_nan,first_best)
    stored_score=tl.where(first_nan<2147483647,float("nan"),maximum)
    tl.store(S+row*T+tile,stored_score)
    tl.store(I+row*T+tile,chosen)

@triton.jit
def _finish(S,I,OUT,T:tl.constexpr,BLOCK:tl.constexpr):
    row=tl.program_id(0)
    t=tl.arange(0,BLOCK)
    valid=t<T
    score=tl.load(S+row*T+t,valid,other=-float("inf"))
    idx=tl.load(I+row*T+t,valid,other=2147483647)
    nan=valid & (score!=score)
    first_nan=tl.min(tl.where(nan,idx,2147483647),0)
    maximum=tl.max(tl.where(valid & ~nan,score,-float("inf")),0)
    first_best=tl.min(tl.where(valid & ~nan & (score==maximum),idx,2147483647),0)
    result=tl.where(first_nan<2147483647,first_nan,first_best)
    tl.store(OUT+row,result.to(tl.int64))

def sample_with_exponential_noise(probs,noise):
    assert probs.ndim==noise.ndim==2 and probs.shape==noise.shape
    assert probs.dtype==noise.dtype==torch.float32
    assert probs.device==noise.device and probs.is_cuda
    batch,n=probs.shape
    assert 0<n<2**31
    output=torch.empty((batch,),device=probs.device,dtype=torch.int64)
    if batch==0:return output
    tiles=triton.cdiv(n,4096)
    scores=torch.empty((batch,tiles),device=probs.device,dtype=torch.float32)
    indices=torch.empty((batch,tiles),device=probs.device,dtype=torch.int32)
    _partial[(tiles,batch)](probs,noise,scores,indices,n,*probs.stride(),*noise.stride(),
                          tiles,4096,num_warps=4,num_stages=1)
    _finish[(batch,)](scores,indices,output,tiles,triton.next_power_of_2(tiles),
                     num_warps=4,num_stages=1)
    return output
