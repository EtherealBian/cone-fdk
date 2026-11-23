import numpy as np
from denoise import Denoise
from pore_check import ROOT


def test_cuda_patch_filter_matches_cpu_at_edges():
    a=np.array([[[3.,7.,8.,1.],[4.,2.,11.,-1.],[0.,.5,4.,6.]]],dtype='f4'); h=3.5
    expected=np.empty_like(a);p=np.pad(a[0],12,mode='edge')
    for y,x in np.ndindex(a.shape[1:]):
        Y,X=y+12,x+12;target=p[Y-2:Y+3,X-2:X+3];weights=[];vals=[]
        for j in range(-10,11):
            for i in range(-10,11):
                neighbor=p[Y+j-2:Y+j+3,X+i-2:X+i+3]
                weights.append(np.exp(-np.mean((target-neighbor)**2)/h**2));vals.append(p[Y+j,X+i])
        expected[0,y,x]=np.average(vals,weights=weights)
    with Denoise(ROOT/'build/pores/nlm.ptx') as g:
        result=g(a,h);np.testing.assert_allclose(result,expected,atol=2e-5,rtol=2e-5)
