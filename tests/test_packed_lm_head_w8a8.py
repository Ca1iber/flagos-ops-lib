"""Independent rank-one full-output reference and caller-buffer contracts."""
import pytest
import torch
from flag_gems.fused.packed_lm_head_w8a8 import pack_lm_head_weight_int8,packed_lm_head_w8a8

torch.set_num_threads(4)

def buffers(m,n):
 qraw=torch.full((32*m*64+256,),13,device='cuda',dtype=torch.int8)
 sraw=torch.full((m+256,),13.,device='cuda')
 yraw=torch.full((m*n+256,),13.,device='cuda',dtype=torch.bfloat16)
 return qraw,sraw,yraw,qraw[128:-128].view(32,m,64),sraw[128:-128],yraw[128:-128].view(m,n)

def guarded(raw):
 assert bool((raw[:128].cpu()==13).all() and (raw[-128:].cpu()==13).all())

@pytest.mark.parametrize('m,n',[(1,129),(7,257),(32,129),(64,130560)])
@torch.inference_mode()
def test_rank_one_full_reference_and_mutable_graph(m,n):
 padded=(n+127)//128*128
 v=(torch.arange(2048,dtype=torch.int32)*3)%17-8
 coeff=torch.arange(n,dtype=torch.int32)%3-1
 w=torch.zeros((padded,2048),dtype=torch.int8)
 w[:n].copy_((coeff[:,None]*v[None,:]).to(torch.int8))
 cpu_packed=w.view(padded//128,128,64,32).permute(0,2,1,3).contiguous()
 packed=cpu_packed.cuda().contiguous()
 sc=(1+torch.arange(n)%11).float()/1024;scales=sc.cuda()
 x=torch.zeros((m,2048),device='cuda',dtype=torch.bfloat16)
 qr,sr,yr,q,s,y=buffers(m,n)
 def fn():return packed_lm_head_w8a8(x,packed,scales,input_quant_workspace=q,input_scale_workspace=s,out=y)
 stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
 with torch.cuda.stream(stream):
  for _ in range(3):fn()
 stream.synchronize();graph=torch.cuda.CUDAGraph()
 with torch.cuda.graph(graph,stream=stream):fn()
 for seed in (35001,35002):
  torch.manual_seed(seed);x.copy_(torch.randint(-15,16,x.shape,device='cuda').to(torch.bfloat16));x[:,0]=15;x[:,1]=-15
  snapshot=x.cpu();graph.replay();torch.cuda.synchronize()
  qs=q.cpu().permute(1,0,2).reshape(m,2048);ss=s.cpu()
  u=snapshot.float()/ss[:,None]
  expected_q=(torch.floor(u.abs()+.5)*torch.where(u<0,-1.,1.)).clamp(-127,127).to(torch.int8)
  assert torch.equal(qs,expected_q)
  dot=(expected_q.to(torch.int64)*v.to(torch.int64)[None,:]).sum(-1)
  ref=((dot[:,None]*coeff.to(torch.int64)[None,:]).float()*ss[:,None]*sc[None,:]).bfloat16()
  torch.testing.assert_close(y.cpu(),ref,rtol=0,atol=0)
  assert torch.equal(x.cpu(),snapshot)
  for raw in (qr,sr,yr):guarded(raw)
 assert torch.equal(packed.cpu(),cpu_packed) and torch.equal(scales.cpu(),sc)

@torch.inference_mode()
def test_weight_codec_zero_padding_and_nonfinite():
 w=((torch.arange(5*2048,device='cuda').reshape(5,2048)%31)-15).bfloat16();w[0].zero_();snapshot=w.cpu()
 q,s=pack_lm_head_weight_int8(w)
 logical=q.cpu().permute(0,2,1,3).reshape(128,2048);sc=s.cpu()
 assert bool((logical[5:]==0).all() and (logical[0]==0).all() and sc[0]==1)
 assert bool(((logical[:5].float()*sc[:,None]-snapshot.float()).abs()<=.5005*sc[:,None]+1e-6).all())
 assert torch.equal(w.cpu(),snapshot)
 bad=w.clone();bad[1,0]=float('nan')
 with pytest.raises(ValueError,match='finite'):pack_lm_head_weight_int8(bad)

@torch.inference_mode()
def test_reject_alias_shape_and_dtype():
 w=torch.zeros((129,2048),device='cuda',dtype=torch.bfloat16);packed,sc=pack_lm_head_weight_int8(w)
 x=torch.zeros((1,2048),device='cuda',dtype=torch.bfloat16);qr,sr,yr,q,s,y=buffers(1,129)
 def call(value=x,weight_scales=sc,qbuf=q,out=y):
  return packed_lm_head_w8a8(value,packed,weight_scales,input_quant_workspace=qbuf,input_scale_workspace=s,out=out)
 with pytest.raises(ValueError):call(out=x[:,:129])
 shared=x.view(torch.int8).flatten()[:2048].view(32,1,64)
 with pytest.raises(ValueError):call(qbuf=shared)
 with pytest.raises(ValueError):call(weight_scales=sc.double())
 with pytest.raises(ValueError):call(value=torch.empty((65,2048),device='cuda',dtype=torch.bfloat16))
 with pytest.raises(ValueError):call(value=torch.empty((0,2048),device='cuda',dtype=torch.bfloat16))
 with pytest.raises(ValueError):call(value=torch.empty((1,4096),device='cuda',dtype=torch.bfloat16)[:,::2])
 for raw in (qr,sr,yr):guarded(raw)

@torch.inference_mode()
def test_optional_output_graph_lifetime():
 w=torch.ones((129,2048),device='cuda',dtype=torch.bfloat16);packed,sc=pack_lm_head_weight_int8(w)
 x=torch.zeros((1,2048),device='cuda',dtype=torch.bfloat16);q=torch.empty((32,1,64),device='cuda',dtype=torch.int8);s=torch.empty(1,device='cuda');holder={}
 def fn():holder['out']=packed_lm_head_w8a8(x,packed,sc,input_quant_workspace=q,input_scale_workspace=s)
 stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
 with torch.cuda.stream(stream):
  for _ in range(3):fn()
 stream.synchronize();graph=torch.cuda.CUDAGraph()
 with torch.cuda.graph(graph,stream=stream):fn()
 x.fill_(1);graph.replay();torch.cuda.synchronize()
 assert holder['out'].shape==(1,129)
 torch.testing.assert_close(holder['out'],torch.full_like(holder['out'],2048),rtol=0,atol=0)
