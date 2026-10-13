# Copyright 2026 FlagOS Contributors
"""Exact bounds and edge coverage for two independent packed radix counts."""
import pytest
import torch
import flag_gems


@pytest.mark.parametrize("n", [1,31,1024,2048])
@pytest.mark.parametrize("pattern", ["low","high","alternate","mixed"])
def test_two_bin_word_has_no_field_carry(n, pattern):
    keys=[0 if pattern=="low" else 1 if pattern=="high" else
          i%2 if pattern=="alternate" else i%16 for i in range(n)]
    total=0
    c0=c1=0
    for key in keys:
        weight=int(key==0) | (int(key==1)<<16)
        total+=weight
        prefix=total-weight
        assert (prefix&65535)==c0
        assert (prefix>>16)==c1
        c0+=int(key==0)
        c1+=int(key==1)
    assert total < 2**32
    assert (total&65535)==c0
    assert (total>>16)==c1


@pytest.mark.sort_stable
@pytest.mark.parametrize("n", [2048,2049,4097])
@pytest.mark.parametrize("descending", [False,True])
def test_two_bin_rank_full_tile_bound(n, descending):
    bits=torch.empty((3,n),dtype=torch.int32,device=flag_gems.device)
    bits[0].fill_(0x3f800000)
    bits[1].fill_(0x3f800001)
    bits[2]=0x3f800000+(torch.arange(n,device=bits.device)%2).to(torch.int32)
    x=bits.view(torch.float32)
    ref_values,ref_indices=torch.sort(x,stable=True,descending=descending)
    with flag_gems.use_gems():
        values,indices=torch.sort(x,stable=True,descending=descending)
    assert indices.dtype==torch.int64
    assert torch.equal(values.view(torch.int32),ref_values.view(torch.int32))
    assert torch.equal(indices,ref_indices)
