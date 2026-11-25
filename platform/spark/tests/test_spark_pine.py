"""Detector preprocessing, archive integrity and absent-source gates. No synthetic physical claim."""
import hashlib
import io
import json
import sys
import zipfile
from contextlib import ExitStack
from pathlib import Path

import numpy as np
from PIL import Image
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from prepare_pine import pine, run, sealed_original


def test_published_scanner_metadata_distinct_from_walnut():
    # The metadata is the publisher's small immutable original, not the projection archive.
    meta=pine.ROOT/'data/pine-cone-originals'/pine.META
    if not meta.is_file():pytest.skip('publisher metadata not staged')
    raw=meta.read_bytes(); assert hashlib.sha256(raw).hexdigest()==pine.source()['acquisition_metadata_sha256']
    assert pine.geometry(raw)==(.05,230.66,553.74)
    with pytest.raises(ValueError):pine.geometry(raw.replace(b'230.66',b'230.67'))


def test_detector_counting_order_keeps_bright_negative_and_zero_floor():
    y,x=np.indices((160,192));a=(4100+3*x+7*y).astype('u2');a[:8,:8]=0;a[-8:,-8:]=65535
    buf=io.BytesIO();Image.fromarray(a).save(buf,format='TIFF');raw=pine.frame(buf.getvalue());assert np.array_equal(a,raw)
    v,air=pine.intensity(raw);v2,air2=pine.intensity(raw,True); assert v.tobytes()==v2.tobytes() and air==air2
    for row,col in [(0,0),(3,14),(19,23)]:
        block=float(sum(int(x) for x in a[row*8:row*8+8,col*8:col*8+8].ravel()))/64
        truth=np.float32(-np.log(max(block,.5)/air));assert v[row,col]==truth
    assert v[-1,-1]<0<v[0,0] and v.shape==(20,24)


@pytest.mark.parametrize('bad',['zero-air','wrong-geometry','unsupported-format'])
def test_reject_detector(bad):
    if bad=='zero-air':
        with pytest.raises(ValueError):pine.intensity(np.zeros((128,128),'u2'))
    elif bad=='wrong-geometry':
        with pytest.raises(ValueError):pine.intensity(np.zeros((135,128),'u2'))
    else:
        b=io.BytesIO();Image.fromarray(np.ones((128,128),'u1')).save(b,format='PNG')
        with pytest.raises(ValueError):pine.frame(b.getvalue())


@pytest.mark.parametrize('mode',['valid','omit','duplicate','outside','unexpected-directory'])
def test_index_exactly_721_frames_without_extraction(tmp_path,mode):
    z1=tmp_path/'one.zip';z2=tmp_path/'two.zip'
    with zipfile.ZipFile(z1,'x') as a,zipfile.ZipFile(z2,'x') as b:
        for k in range(1,722):
            if k==96 and mode=='omit':continue
            (a if k<=350 else b).writestr(f'20201118_pine_cone_{k:04}.tif',b'synthetic')
        if mode=='duplicate':b.writestr('20201118_pine_cone_0007.tif',b'x')
        if mode=='outside':a.writestr('20201118_pine_cone_0722.tif',b'x')
        if mode=='unexpected-directory':a.writestr('else/20201118_pine_cone_0001.tif',b'x')
    with zipfile.ZipFile(z1) as a,zipfile.ZipFile(z2) as b:
        if mode=='valid':
            r=pine.index({'one':a,'two':b}); assert set(r)==set(range(1,722))
            assert pine.member(b,r[721][2])==b'synthetic'
        else:
            with pytest.raises(ValueError):pine.index({'one':a,'two':b})


@pytest.mark.parametrize('wrong',['receipt','publisher','length'])
def test_input_requires_both_checksums_and_size(tmp_path,wrong):
    f=tmp_path/'archive'; b=b'fixed bytes'; f.write_bytes(b); pin=hashlib.sha256(b).hexdigest()
    x={'bytes':len(b),'md5':hashlib.md5(b).hexdigest()}
    if wrong=='receipt':pin='0'*64
    if wrong=='publisher':x['md5']='0'*32
    if wrong=='length':x['bytes']+=1
    with pytest.raises(ValueError):
        with sealed_original(f,pin,x) as stream:pine.full_file(stream,x)


def test_incomplete_source_cannot_publish_scan(tmp_path):
    orig=tmp_path/'originals';orig.mkdir();dest=tmp_path/'scans'
    with pytest.raises(FileNotFoundError):run(orig,dest)
    assert not dest.exists() and not list(tmp_path.glob('.pine-scan-*'))
