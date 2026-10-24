"""Full-output integer reference, mutable graphs and disjoint buffer guards."""
import pytest
import torch
from flag_gems.fused.packed_projection_w8a8 import pack_projection_weight_int8, packed_projection_w8a8

torch.set_num_threads(4)

@pytest.mark.parametrize('m,n',[(1,2560),(336,2560),(511,2048),(565,2560),(769,2048),(864,2048),(1537,2560),(2048,2048),(2048,2560)])
@torch.inference_mode()
def test_full_rank_one_and_mutable_graph(m,n):
    # W[n,k]=coefficient[n]*v[k]; independent complete dot reduces to one
    # integer vector dot per row, not a reimplementation of device tiling.
    v=(torch.arange(2048,dtype=torch.int32)*3)%17-8
    coeff=torch.arange(n,dtype=torch.int32)%3-1
    w=(coeff[:,None]*v[None,:]).to(torch.int8)
    cpu_packed=w.view(n//128,128,64,32).permute(0,2,1,3).contiguous()
    packed=cpu_packed.cuda().contiguous()
    sc=(1+torch.arange(n)%11).float()/1024;scales=sc.cuda()
    x=torch.zeros((m,2048),device='cuda',dtype=torch.bfloat16)
    qr=torch.full((32*m*64+256,),13,device='cuda',dtype=torch.int8)
    sr=torch.full((m+256,),13.,device='cuda')
    yr=torch.full((m*n+256,),13.,device='cuda',dtype=torch.bfloat16)
    q=qr[128:-128].view(32,m,64);s=sr[128:-128];y=yr[128:-128].view(m,n)
    def fn():return packed_projection_w8a8(x,packed,scales,input_quant_workspace=q,input_scale_workspace=s,out=y)
    stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):fn()
    stream.synchronize();graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph,stream=stream):fn()
    for seed in (39001,39002):
        torch.manual_seed(seed);x.copy_(torch.randint(-15,16,x.shape,device='cuda').to(torch.bfloat16))
        x[:,0]=15;x[:,1]=-15;x[0].zero_()
        snapshot=x.cpu();graph.replay();torch.cuda.synchronize()
        qs=q.cpu().permute(1,0,2).reshape(m,2048);ss=s.cpu()
        expected_s=snapshot.float().abs().amax(-1)/127
        expected_s=torch.where(expected_s>0,expected_s,1.)
        torch.testing.assert_close(ss,expected_s,rtol=1e-6,atol=0)
        u=snapshot.float()/ss[:,None]
        expected_q=(torch.floor(u.abs()+.5)*torch.where(u<0,-1.,1.)).clamp(-127,127).to(torch.int8)
        assert torch.equal(qs,expected_q)
        dot=(expected_q.to(torch.int64)*v.to(torch.int64)[None,:]).sum(-1)
        ref=((dot[:,None]*coeff.to(torch.int64)[None,:]).float()*ss[:,None]*sc[None,:]).bfloat16()
        torch.testing.assert_close(y.cpu(),ref,rtol=0,atol=0)
        assert torch.equal(x.cpu(),snapshot)
        for raw in (qr,sr,yr):
            assert bool((raw[:128].cpu()==13).all() and (raw[-128:].cpu()==13).all())
    assert torch.equal(packed.cpu(),cpu_packed) and torch.equal(scales.cpu(),sc)

@torch.inference_mode()
def test_weight_codec_and_contract():
    w=((torch.arange(2048*2048,device='cuda').reshape(2048,2048)%31)-15).bfloat16()
    w[0].zero_();snapshot=w.cpu();packed,sc=pack_projection_weight_int8(w)
    logical=packed.cpu().permute(0,2,1,3).reshape(2048,2048);s=sc.cpu()
    assert s[0]==1 and bool((logical[0]==0).all())
    assert bool(((logical.float()*s[:,None]-snapshot.float()).abs()<=.5005*s[:,None]+1e-6).all())
    assert torch.equal(w.cpu(),snapshot)
    w[1,0]=float('nan')
    with pytest.raises(ValueError,match='finite'):pack_projection_weight_int8(w)
    with pytest.raises(ValueError):pack_projection_weight_int8(w[:128])

@torch.inference_mode()
def test_reject_alias_bad_shapes_and_optional_output():
    w=torch.ones((2048,2048),device='cuda',dtype=torch.bfloat16)
    packed,sc=pack_projection_weight_int8(w)
    x=torch.ones((1,2048),device='cuda',dtype=torch.bfloat16)
    q=torch.empty((32,1,64),device='cuda',dtype=torch.int8);s=torch.empty(1,device='cuda')
    def call(value=x,weight=packed,scales=sc,qbuf=q,sbuf=s,out=None):
        return packed_projection_w8a8(value,weight,scales,input_quant_workspace=qbuf,input_scale_workspace=sbuf,out=out)
    with pytest.raises(ValueError):call(out=x)
    with pytest.raises(ValueError):call(qbuf=x.view(torch.int8).flatten()[:2048].view(32,1,64))
    with pytest.raises(ValueError):call(sbuf=sc[:1])
    with pytest.raises(ValueError):call(scales=sc.double())
    with pytest.raises(ValueError):call(weight=packed.to(torch.int16))
    for m in (0,2049):
        with pytest.raises(ValueError):call(value=torch.empty((m,2048),device='cuda',dtype=torch.bfloat16))
    with pytest.raises(ValueError):call(value=torch.empty((1,4096),device='cuda',dtype=torch.bfloat16)[:,::2])
    holder={};stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):holder['y']=call()
    stream.synchronize();graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph,stream=stream):holder['y']=call()
    x.zero_();graph.replay();torch.cuda.synchronize()
    torch.testing.assert_close(holder['y'],torch.zeros_like(holder['y']),rtol=0,atol=0)
