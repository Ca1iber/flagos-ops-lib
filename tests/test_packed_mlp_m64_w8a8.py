"""Independent quantized-stage tests; these do not certify model accuracy."""
import pytest
import torch
import flag_gems
pytestmark=pytest.mark.skipif(flag_gems.vendor_name!="metax",reason="MetaX W8A8")
from flag_gems.fused.packed_mlp_m64_w8a8 import pack_gate_up_weight_int8,packed_mlp_m64_w8a8
from flag_gems.fused.packed_bundle import pack_linear_weight

@pytest.fixture(scope='module')
def case():
    torch.manual_seed(20262600)
    with torch.inference_mode():
        weight=torch.randn((12288,2048),device='cuda',dtype=torch.bfloat16)*.02
        weight[0].zero_()
        down=torch.randn((2048,6144),device='cuda',dtype=torch.bfloat16)*.02
        packed,scales=pack_gate_up_weight_int8(weight)
        yield dict(weight=weight,down=down,packed=packed,scales=scales,
                   down_packed=pack_linear_weight(down,panel_n=64,block_k=64))


def buffers():
    return dict(input_quant_workspace=torch.empty((32,64,64),device='cuda',dtype=torch.int8),
                input_scale_workspace=torch.empty((64,),device='cuda',dtype=torch.float32),
                activation_workspace=torch.empty((96,64,64),device='cuda',dtype=torch.bfloat16),
                partial_workspace=torch.empty((4,64,2048),device='cuda',dtype=torch.float32),
                out=torch.empty((64,2048),device='cuda',dtype=torch.bfloat16))


def close(actual,expected):
    torch.testing.assert_close(actual.cpu().float(),expected.cpu().float(),atol=.01,rtol=.02)


def test_weight_codec(case):
    q=case['packed'].cpu().permute(0,2,1,3).reshape(6144,2,2048).permute(1,0,2).reshape(12288,2048)
    s=case['scales'].cpu().view(6144,2).t().contiguous().reshape(12288)
    w=case['weight'].cpu().float()
    assert bool((q.short().abs()<=127).all()) and bool(torch.isfinite(s).all() and (s>0).all())
    assert bool(((w-q.float()*s[:,None]).abs()<=.5005*s[:,None]+1e-6).all())
    assert bool((q[0]==0).all()) and s[0]==1


@torch.inference_mode()
def test_graph_fresh_input_independent_integer_reference(case):
    b=buffers();x=torch.randn((64,2048),device='cuda',dtype=torch.bfloat16)
    def run():return packed_mlp_m64_w8a8(x,case['packed'],case['scales'],case['down_packed'],**b)
    stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):run()
    stream.synchronize();g=torch.cuda.CUDAGraph()
    with torch.cuda.graph(g,stream=stream):run()
    qw=case['packed'].cpu().permute(0,2,1,3).reshape(6144,2,2048).permute(1,0,2).reshape(12288,2048).double()
    sw=case['scales'].cpu().view(6144,2).t().contiguous().reshape(12288)
    dw=case['down'].cpu().float();ids=[0,31,63]
    weight_before=case['packed'].cpu();scale_before=case['scales'].cpu()
    for pattern in ('normal','large','zero'):
        if pattern=='zero':x.zero_()
        else:
            x.normal_()
            if pattern=='large':x.mul_(8)
        source=x.cpu();g.replay();torch.cuda.synchronize()
        qx=b['input_quant_workspace'].cpu().permute(1,0,2).reshape(64,2048)
        sx=b['input_scale_workspace'].cpu()
        assert bool(((source.float()-qx.float()*sx[:,None]).abs()<=.5005*sx[:,None]+1e-6).all())
        projected=((qx[ids].double()@qw.t()).float()*sx[ids,None]*sw[None,:]).bfloat16().float()
        gate,up=projected[:,:6144],projected[:,6144:]
        reference=(gate*torch.sigmoid(gate)*up).bfloat16()
        actual_z=b['activation_workspace'].cpu().permute(1,0,2).reshape(64,6144)
        close(actual_z[ids],reference)
        close(b['out'][ids],(actual_z[ids].float()@dw.t()).bfloat16())
        assert torch.equal(x.cpu(),source)
    assert torch.equal(case['packed'].cpu(),weight_before) and torch.equal(case['scales'].cpu(),scale_before)


@torch.inference_mode()
def test_alias_and_contract_rejection(case):
    x=torch.zeros((64,2048),device='cuda',dtype=torch.bfloat16);b=buffers()
    with pytest.raises(ValueError,match='overlap'):
        packed_mlp_m64_w8a8(x,case['packed'],case['scales'],case['down_packed'],**{**b,'out':x})
    with pytest.raises(ValueError,match='input scale'):
        packed_mlp_m64_w8a8(x,case['packed'],case['scales'],case['down_packed'],**{**b,'input_scale_workspace':b['input_scale_workspace'].view(64,1)})


@torch.inference_mode()
def test_nonfinite_frozen_weights_rejected(case):
    bad=case['weight'].clone();bad[0,0]=float('nan')
    with pytest.raises(ValueError,match='finite'):
        pack_gate_up_weight_int8(bad)
