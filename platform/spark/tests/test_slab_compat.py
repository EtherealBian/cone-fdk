"""Synthetic negative controls; run the facade in a new interpreter, not the pytest import cache."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
import build_slab_compat as builder
from qualify_slab_compat import OVERLAY


@pytest.fixture
def overlay(tmp_path):
    out = tmp_path / "overlay"
    assert builder.build(out, OVERLAY) == OVERLAY
    return out


@pytest.mark.parametrize("source", builder.PINS)
def test_modified_frozen_input_refused(tmp_path, source):
    p = tmp_path / source
    p.parent.mkdir(parents=True)
    p.write_bytes((builder.ROOT / source).read_bytes() + b"# modified\n")
    with pytest.raises(ValueError, match="frozen source changed"):
        builder.checked(tmp_path, source)


def test_unexpected_manifest_does_not_publish(tmp_path):
    target = tmp_path / "product"
    with pytest.raises(ValueError, match="unexpected overlay"):
        builder.build(target, "0" * 64)
    assert not target.exists()


@pytest.mark.parametrize("change", ["content", "symlink", "extra", "absent"])
def test_changed_overlay_refused_without_repair(overlay, change):
    p = overlay / "_spark_slab_core.py"
    if change == "content":
        p.write_bytes(p.read_bytes() + b"# changed\n")
    if change == "symlink":
        other = overlay.parent / "external"
        p.rename(other)
        p.symlink_to(other)
    if change == "extra":
        (overlay / "unlisted").touch()
    if change == "absent":
        p.unlink()
    before = [(q.name, q.lstat().st_ino) for q in overlay.iterdir()]
    with pytest.raises(ValueError, match="not identical"):
        builder.build(overlay, OVERLAY)
    assert before == [(q.name, q.lstat().st_ino) for q in overlay.iterdir()]


CHILD = r"""
import errno,fcntl,os,sys
from pathlib import Path
original,alias,expected=Path(sys.argv[1]),Path(sys.argv[2]),bytes.fromhex(sys.argv[3]);original.rename(original.with_name('old-'+original.name));original.write_bytes(b'X'*len(expected));assert alias.read_bytes()==expected
with alias.open('rb') as f:
    required=fcntl.F_SEAL_SEAL|fcntl.F_SEAL_WRITE|fcntl.F_SEAL_GROW|fcntl.F_SEAL_SHRINK
    assert fcntl.fcntl(f,fcntl.F_GET_SEALS)&required==required
try:
    with alias.open('r+b',buffering=0) as f:f.write(b'X')
except PermissionError as e:assert e.errno in (errno.EPERM,errno.EACCES)
else:raise AssertionError('modified sealed bytes')
assert original.read_bytes()!=expected and alias.read_bytes()==expected
"""
PROBE = r"""
import hashlib,json,os,subprocess,sys
from pathlib import Path
sys.path.insert(0,sys.argv[1]);import slab_replay as sr
root=Path(sys.argv[2]);index=int(sys.argv[3]);child=sys.argv[4];sc=root/'source';b=root/'bin';b.mkdir();files=sr.paths(sc,b)
expected=[b'geometry',b'0123456789ab',b'ELF fixture',b'PTX fixture'];vol=bytes(range(108));sha=lambda b:hashlib.sha256(b).hexdigest()
for p,v in zip(files,expected,strict=True):p.write_bytes(v)
d=dict(format='CFDK-slab-replay-plan-v1',n=3,fov_mm=6.,batch=1,budget_mib=3,projections_bytes=12,geometry_sha256=sha(expected[0]),projections_sha256=sha(expected[1]),products={n:sha(x) for n,x in zip(sr.NAMES,expected[2:],strict=True)},volume_sha256=sha(vol),slabs=[dict(z=0,depth=3,sha256=sha(vol))]);plan=root/'plan';plan.write_bytes(sr.encode(d));args=(plan,sha(plan.read_bytes()),sc,b,root/'cache',root/'delivery');seen=[]
def worker(scan,build,raw,d,s):
    aliases=sr.paths(scan,build);seen.extend(aliases);assert len(set(aliases))==4
    if index==-1:
        aliases[0].unlink();aliases[0].read_bytes();raise AssertionError('missing view was reopened from source')
    subprocess.run([sys.executable,'-I','-S','-B','-c',child,str(files[index]),str(aliases[index]),expected[index].hex()],check=True,close_fds=True,timeout=10)
    assert [p.read_bytes() for p in aliases]==expected
    raw.write_bytes(vol);return dict(gpu='analytic control',driver_api=1,n=3,batch=1,roi_origin_xyz=[0,0,0],output_shape_xyz=[3,3,3],voxel_spacing_mm=2,redzones_verified=True,peak_owned_bytes=16,budget_bytes=3<<20,output_bytes=len(vol))
before=len(os.listdir('/proc/self/fd'))
if index==-1:
    try:sr.generate(*args,native=worker)
    except FileNotFoundError:pass
    else:raise AssertionError('missing alias not refused')
    assert not args[-1].exists()
else:
    r=sr.generate(*args,native=worker);assert r['computed']==1 and (args[-1]/'volume.f32').read_bytes()==vol
    try:sr.generate(*args[:-1],root/'other',native=worker)
    except ValueError as e:assert 'changed' in str(e)
    else:raise AssertionError('changed source accepted on resume')
    assert not (root/'other').exists()
assert seen and all(not os.path.lexists(p) for p in seen) and before==len(os.listdir('/proc/self/fd'))
"""


@pytest.mark.parametrize(
    "index",
    [-1, 0, 1, 2, 3],
    ids=["removed-alias", "geometry", "projections", "executable", "ptx"],
)
def test_sealed_legacy_paths_in_external_process(overlay, tmp_path, index):
    subprocess.run(
        [
            sys.executable,
            "-I",
            "-S",
            "-B",
            "-c",
            PROBE,
            str(overlay),
            str(tmp_path),
            str(index),
            CHILD,
        ],
        check=True,
        timeout=30,
    )
