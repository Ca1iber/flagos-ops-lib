# Copyright 2026 FlagOS Contributors
"""Exact adjacent-digit identity flags and mixed-row replay regression."""
import pytest
import torch
import flag_gems


@pytest.mark.parametrize("bits",[32,64])
def test_digit_or_predicate_is_elementwise_exact(bits):
    words=[0,1,0x10,0x01,0xffff0000,0x12344444,(1<<bits)-1]
    mismatch=0
    for word in words:mismatch|=word^(word>>4)
    for p in range(1,bits//4):
        expected=all(((word>>(4*p))&15)==((word>>(4*(p-1)))&15) for word in words)
        assert (((mismatch>>(4*(p-1)))&15)==0)==expected


@pytest.mark.sort_stable
@pytest.mark.parametrize("descending",[False,True])
def test_repeat_digit_mixed_rows(descending):
    x=torch.randn((8,130560),device=flag_gems.device)
    x[:4]=x[:4].bfloat16().float()
    x[4:6]=x[4:6].half().float()
    refv,refi=torch.sort(x,stable=True,descending=descending)
    with flag_gems.use_gems():
        v,i=torch.sort(x,stable=True,descending=descending)
    assert torch.equal(v,refv) and torch.equal(i,refi)
    assert i.dtype==torch.int64


@pytest.mark.sort_stable
@pytest.mark.skipif(flag_gems.device!="cuda",reason="CUDA-alike graph API")
def test_repeat_digit_mixed_rows_graph():
    x=torch.randn((8,4097),device=flag_gems.device)
    stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):flag_gems.sort(x)
    torch.cuda.current_stream().wait_stream(stream)
    graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):v,i=flag_gems.sort(x)
    for rounded_rows in [0,4,8]:
        x.normal_()
        x[:rounded_rows]=x[:rounded_rows].bfloat16().float()
        graph.replay();torch.cuda.synchronize()
        refv,refi=torch.sort(x,stable=True)
        assert torch.equal(v,refv) and torch.equal(i,refi)
