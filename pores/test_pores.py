import subprocess

import numpy as np
import pytest
from pore_check import command, native, reference
from scipy import ndimage as ndi


def case(kind):
    rng=np.random.default_rng(144)
    if kind=='mixed': return rng.integers(-2,5,(11,37,47)).astype('f4')
    if kind=='empty': return np.ones((1,1,1),'f4')
    if kind=='full': return np.zeros((9,1,43),'f4')
    a=np.ones((13,19,31),'f4'); a[:,3,2]=0
    for z in range(13):
        a[z, z+1, 10]=0  # edge adjacency
        a[z, z+1, z+16]=0 # corner adjacency
    a[2,8,2:17]=0; a[3:10,8,2]=0; a[3:10,8,16]=0; a[10,8,2:17]=0
    return a


@pytest.mark.parametrize('kind',['mixed','empty','full','seams'])
@pytest.mark.parametrize('conn',[6,26])
def test_oracle_and_partition(tmp_path,kind,conn):
    a=case(kind); src=tmp_path/'测试.f32'; a.tofile(src); spacing=(0.7,2.,3.5); baseline=None
    for slab in (1,4,63):
        out=tmp_path/f'{slab}.csv'; stats=native(src,out,a.shape,spacing,0,conn,slab)
        reference(a<=0,conn,spacing,out,stats)
        if baseline is None: baseline=out.read_bytes()
        assert out.read_bytes()==baseline
        assert stats['device_buffer_bytes']==8*a[0].size*min(slab,len(a))


def test_median_xy(tmp_path):
    a=case('mixed'); src=tmp_path/'raw'; a.tofile(src); out=tmp_path/'out'; spacing=(2,3,5)
    st=native(src,out,a.shape,spacing,1,26,3,'median-xy'); mask=ndi.median_filter(a,size=(1,3,3),mode='nearest')<=1
    reference(mask,26,spacing,out,st)


@pytest.mark.parametrize('failure',['nan-last','inf','short','long','negative-spacing','budget','alias','conn','filter','repeat'])
def test_atomic_failure(tmp_path,failure):
    shape=(3,9,11) if failure!='budget' else (1,512,512)
    a=np.zeros(shape,'f4'); src=tmp_path/'raw'; out=tmp_path/'out'; out.write_bytes(b'sentinel')
    if failure=='nan-last': a.flat[-1]=np.nan
    if failure=='inf': a.flat[0]=-np.inf
    a.tofile(src)
    if failure=='short': src.write_bytes(src.read_bytes()[:-1])
    if failure=='long': src.write_bytes(src.read_bytes()+b'.')
    before=src.read_bytes(); cmd=command(src,src if failure=='alias' else out,shape,(-1 if failure=='negative-spacing' else 2,3,5),0,12 if failure=='conn' else 6,1,'nonsense' if failure=='filter' else 'none',1)
    if failure=='repeat': cmd.extend(['--nx','11'])
    assert subprocess.run(cmd,capture_output=True).returncode!=0
    assert src.read_bytes()==before and out.read_bytes()==b'sentinel' and not list(tmp_path.glob('*.partial.*'))
