"""Qualifier controls, including a native/ASTRA analytic smoke test; not measured pine data."""
import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import qualify_pine as q
from common import Geometry
from phantom import generate


def valid():
    return dict(gpu='NVIDIA GB10',views=90,n=64,requested_batch=16,batch=16,budget_bytes=16<<20,
        redzones_verified=True,peak_owned_bytes=2048,effective_budget_bytes=16<<20,
        roi_origin_xyz=[0,0,0],output_shape_xyz=[64]*3,output_bytes=4*64**3,
        voxel_spacing_mm=.125,projection_passes=1,projection_bytes_read=4096)


@pytest.mark.parametrize('key,value',[('redzones_verified',False),('views',720),('batch',12),('budget_bytes',32<<20),('peak_owned_bytes',(16<<20)+1),('roi_origin_xyz',[0,1,0]),('projection_bytes_read',0),('output_bytes',0)])
def test_refuse_false_device_receipt(key,value):
    r=valid();q.guards(r,64,90,8,16,16,[0,0,0,64,64,64],4096);r[key]=value
    with pytest.raises(ValueError,match='accounting'):q.guards(r,64,90,8,16,16,[0,0,0,64,64,64],4096)


def test_axis_order_and_single_byte_damage(tmp_path):
    f=np.arange(7*8*9,dtype='<f4').reshape(7,8,9);p=tmp_path/'roi';f[2:5,1:5,3:8].tofile(p)
    assert q.check_roi(f,p,[3,1,2,5,4,3])==q.sha(p)
    b=bytearray(p.read_bytes());b[-1]^=1;p.write_bytes(b)
    with pytest.raises(ValueError,match='differs'):q.check_roi(f,p,[3,1,2,5,4,3])


@pytest.mark.parametrize('damage',['blank','nonfinite','short'])
def test_invalid_volume(tmp_path,damage):
    p=tmp_path/'out';v=np.zeros((3,2,4),'<f4')
    if damage=='nonfinite':v[0]=np.nan
    v.tofile(p)
    if damage=='short':p.write_bytes(p.read_bytes()[:-4])
    with pytest.raises(ValueError):q.volume(p,(3,2,4))


def test_cannot_publish_an_incomplete_or_forged_scan(tmp_path):
    s=tmp_path/'scans';s.mkdir();p=s/'scan.json';p.write_text('{"format":"not-a-complete-scan"}')
    with pytest.raises((ValueError,KeyError)):q.provenance(s,q.sha(p),tmp_path/'receipt')
    with pytest.raises(ValueError):q.provenance(s,'0'*64,tmp_path/'receipt')
    assert list(s.iterdir())==[p]


def test_pine_measurement_adapter_on_analytic_payload(tmp_path):
    # Exercise FD access in the real executable and ASTRA on a non-measured input.
    # Reuse only the synthetic projection bytes. Nominal pine geometry exercises the binding;
    # this intentionally does not issue a physical qualification or call the production run().
    b=q.ROOT/'build/spark'; rec=q.checked(b/'build.json',q.BUILD_PIN,1<<16)
    src=tmp_path/'pine90'; generate(src,90,64,80); Geometry(90,64,80,.4,.4,230.66,553.74).save(src.with_suffix('.geom'))
    meta=dict(views=90,shape_vu=[64,80],projection_bytes=src.with_suffix('.f32').stat().st_size,
              geom_sha256=q.sha(src.with_suffix('.geom')),projections_sha256=q.sha(src.with_suffix('.f32')))
    bound=tmp_path/'sealed';bound.mkdir();out=tmp_path/'synthetic';settings=dict(n=64,fov_mm=8.,batch=16,budget_mib=16,rois_xyzwhd=[[4,6,8,9,11,13],[28,30,32,7,9,5]])
    with q.inputs(tmp_path,b,meta,'pine90',bound,rec['products']) as scan:
        src.with_suffix('.geom').write_text('intentionally changed original');src.with_suffix('.f32').unlink()
        result=q.measure(scan,bound,out,90,meta,settings)
    assert result['native_vs_astra']['rel_l2']<.002 and len(result['rois'])==2
    assert result['dense_sha256']==result['repeat_sha256']==q.sha(out/'native-0.f32') and not (out/'native-1.f32').exists()
    assert not (out/'qualification.json').exists()
