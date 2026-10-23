"""API wiring, masked tails, ownership and mutable Graph replay on MetaX."""
import importlib
import pytest
import torch
from flag_gems import runtime
A=importlib.import_module('flag_gems.fused.packed_mlp_w8a8')
G=importlib.import_module('flag_gems.fused.packed_bundle')
pytestmark=pytest.mark.skipif(not torch.cuda.is_available() or runtime.device.vendor_name!='metax',reason='MetaX device required')

@pytest.fixture(scope='module')
def weights():
    # Known rank-one integer matrix gives an independent full-output reference.
    # n is interleaved gate/up position; k pattern does not divide K32/K64.
    n=torch.arange(12288,dtype=torch.int32)
    v=((torch.arange(2048,dtype=torch.int32)*3)%17-8).to(torch.int8)
    coeff=(n%3-1).to(torch.int8)
    q=coeff[:,None]*v[None,:]
    packed_cpu=q.reshape(96,128,64,32).permute(0,2,1,3).contiguous()
    scale_cpu=(1+n%11).float()/1024
    # Each output touches all three separated thirds of Z.
    dw=torch.zeros((2048,6144),dtype=torch.bfloat16)
    i=torch.arange(2048);dw[i,i]=1;dw[i,i+2048]=.5;dw[i,i+4096]=-.25
    dp_cpu=G.pack_linear_weight(dw,panel_n=64,block_k=64)
    return dict(q=packed_cpu.cuda().contiguous(),s=scale_cpu.cuda(),d=dp_cpu.cuda().contiguous(),
                packed_cpu=packed_cpu,scale_cpu=scale_cpu,down_cpu=dp_cpu,coeff=coeff,v=v)

def buffer(shape,dtype):
    count=1
    for n in shape:count*=n
    raw=torch.full((count+256,),13,device='cuda',dtype=dtype)
    return raw,raw[128:-128].view(shape)

def args(m):
    tensors=[buffer((m,2048),torch.bfloat16),buffer((32,m,64),torch.int8),
             buffer((m,),torch.float32),buffer((96,m,64),torch.bfloat16),
             buffer((m,2048),torch.bfloat16)]
    x,q,s,z,y=[t[1] for t in tensors]
    return tensors,x,q,s,z,y

def run(w,x,q,s,z,y):
    return A.packed_mlp_w8a8(x,w['q'],w['s'],w['d'],input_quant_workspace=q,
        input_scale_workspace=s,activation_workspace=z,out=y)

def check_reference(w,x,q,s,z,y):
    m=x.shape[0]
    xc=x.cpu().float();qx=q.cpu().permute(1,0,2).reshape(m,2048);sx=s.cpu()
    assert bool(torch.isfinite(sx).all() and (sx>0).all())
    assert bool((qx.short().abs()<=127).all())
    assert bool(((xc-qx.float()*sx[:,None]).abs()<=.5005*sx[:,None]+1e-6).all())
    # Exact integer sum, independently derived from rank-one logical weights.
    isum=(qx.long()*w['v'].long()[None,:]).sum(1)
    dot=isum[:,None]*w['coeff'].long()[None,:]
    projected=(dot.float()*sx[:,None]*w['scale_cpu'][None,:]).bfloat16().float()
    pair=projected.reshape(m,6144,2);gate,up=pair[...,0],pair[...,1]
    expected_z=(gate*torch.sigmoid(gate)*up).bfloat16()
    actual_z=z.cpu().permute(1,0,2).reshape(m,6144)
    torch.testing.assert_close(actual_z,expected_z,atol=.01,rtol=.02)
    # Check complete independent coupled result and actual-Z down wiring.
    expected_y=(expected_z[:,:2048].float()+.5*expected_z[:,2048:4096].float()-.25*expected_z[:,4096:].float()).bfloat16()
    actual_down=(actual_z[:,:2048].float()+.5*actual_z[:,2048:4096].float()-.25*actual_z[:,4096:].float()).bfloat16()
    torch.testing.assert_close(y.cpu(),expected_y,atol=.01,rtol=.02)
    torch.testing.assert_close(y.cpu(),actual_down,atol=.01,rtol=.02)
    if bool((xc==0).all()):
        assert bool((sx==1).all() and (qx==0).all() and (y.cpu()==0).all())

@pytest.mark.parametrize('m',[1,72,565,1537,2048])
def test_full_reference_and_graph_updates(weights,m):
    with torch.inference_mode():
        storage,x,q,s,z,y=args(m);torch.manual_seed(2900+m);x.normal_()
        assert run(weights,x,q,s,z,y).data_ptr()==y.data_ptr()
        st=torch.cuda.Stream();st.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(st):
            for _ in range(3):run(weights,x,q,s,z,y)
        st.synchronize();graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph,stream=st):run(weights,x,q,s,z,y)
        for pattern in ('zero','fresh'):
            if pattern=='zero':x.zero_()
            else:x.normal_().mul_(3)
            before=x.cpu();graph.replay();torch.cuda.synchronize()
            assert torch.equal(before,x.cpu())
            check_reference(weights,x,q,s,z,y)
        for raw,_ in storage:
            assert bool((raw[:128].cpu()==13).all() and (raw[-128:].cpu()==13).all())
        assert torch.equal(weights['q'].cpu(),weights['packed_cpu'])
        assert torch.equal(weights['s'].cpu(),weights['scale_cpu'])
        assert torch.equal(weights['d'].cpu(),weights['down_cpu'])


def test_alias_shape_and_dtype_guards(weights):
    with torch.inference_mode():
        _,x,q,s,z,y=args(72);x.normal_()
        with pytest.raises(ValueError):run(weights,x,q,s,z,x)
        with pytest.raises(ValueError):run(weights,x,q,s,z,z.flatten()[:72*2048].view(72,2048))
        alias=weights['q'].flatten()[:72*2048].view(32,72,64)
        with pytest.raises(ValueError):run(weights,x,alias,s,z,y)
        with pytest.raises(ValueError):run(weights,x,q,s[:-1],z,y)
        with pytest.raises((ValueError,TypeError)):run(weights,x.float(),q,s,z,y)
        with pytest.raises(ValueError):run(weights,x[:0],q,s,z,y)
        too_large=torch.empty((2049,2048),device='cuda',dtype=torch.bfloat16)
        with pytest.raises(ValueError):run(weights,too_large,q,s,z,y)
        noncontiguous=torch.empty((72,4096),device='cuda',dtype=torch.bfloat16)[:,::2]
        with pytest.raises(ValueError):run(weights,noncontiguous,q,s,z,y)


def test_reuse_bf16_scratch_without_extra_quant_allocation(weights):
    with torch.inference_mode():
        m=1537;_,x,_,s,z,y=args(m);x.normal_()
        scratch=torch.full((m*2048,),13.,device='cuda',dtype=torch.bfloat16)
        raw=scratch.view(torch.int8)
        q=raw[:m*2048].view(32,m,64);tail=raw[m*2048:].clone()
        assert q.data_ptr()==scratch.data_ptr()
        run(weights,x,q,s,z,y);torch.cuda.synchronize()
        check_reference(weights,x,q,s,z,y)
        assert torch.equal(raw[m*2048:],tail)


def test_optional_output_graph_lifetime(weights):
    with torch.inference_mode():
        _,x,q,s,z,_=args(96);x.normal_()
        def allocate():
            return A.packed_mlp_w8a8(x,weights['q'],weights['s'],weights['d'],
                input_quant_workspace=q,input_scale_workspace=s,activation_workspace=z)
        st=torch.cuda.Stream();st.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(st):
            for _ in range(3):allocate()
        st.synchronize();graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph,stream=st):y=allocate()
        assert tuple(y.shape)==(96,2048) and y.dtype==torch.bfloat16
        for zero in (True,False):
            if zero:x.zero_()
            else:x.normal_()
            graph.replay();torch.cuda.synchronize();check_reference(weights,x,q,s,z,y)
