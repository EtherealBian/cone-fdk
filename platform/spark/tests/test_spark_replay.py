"""Platform-specific failure semantics; these do not pretend Linux locks original names."""
from contextlib import closing
import errno
import fcntl
import hashlib
import os
import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import replay_spark as replay
from sealed_inputs import SEALS, publish_new


def sha(b):return hashlib.sha256(b).hexdigest()


@pytest.fixture
def case(tmp_path):
    b=tmp_path/'build'; b.mkdir(); scan=tmp_path/'sample'; files=replay.sr.paths(scan,b)
    for i,p in enumerate(files):p.write_bytes(bytes([50+i])*(12 if i==1 else 14))
    vol=bytes(range(108)); h=[sha(p.read_bytes()) for p in files]
    plan=dict(format='CFDK-slab-replay-plan-v1',n=3,fov_mm=6.,batch=1,budget_mib=3,
              projections_bytes=12,geometry_sha256=h[0],projections_sha256=h[1],
              products=dict(zip(replay.sr.NAMES,h[2:],strict=True)),volume_sha256=sha(vol),
              slabs=[dict(z=0,depth=1,sha256=sha(vol[:36])),dict(z=1,depth=2,sha256=sha(vol[36:]))]); pf=tmp_path/'plan';pf.write_bytes(replay.sr.encode(plan)); args=[pf,sha(pf.read_bytes()),scan,b,tmp_path/'cache',tmp_path/'delivery']
    def worker(fds,out,d,s):
        z=s['z'];n=replay.sr.size(d,s);out.write_bytes(vol[z*36:z*36+n]);return dict(gpu='synthetic only',driver_api=0,n=3,batch=1,voxel_spacing_mm=2.,roi_origin_xyz=[0,0,z],output_shape_xyz=[3,3,s['depth']],redzones_verified=True,peak_owned_bytes=32,budget_bytes=3<<20,output_bytes=n)
    return args,files,worker,plan


@pytest.mark.parametrize('i',range(4))
def test_snapshot_independent_of_original_and_cannot_be_written(case,i):
    args,files,_,d=case; orig=files[i].read_bytes()
    with replay.sealed_scan(args[2],args[3],d) as fds:
        fd=fds[i];assert fcntl.fcntl(fd,fcntl.F_GET_SEALS)&SEALS==SEALS
        files[i].rename(files[i].with_name('old-'+files[i].name));files[i].write_bytes(b'new')
        assert subprocess.check_output([sys.executable,'-c',f'import os;print(os.pread({fd}, 60, 0).hex())'],pass_fds=(fd,)).strip()==orig.hex().encode()
        with pytest.raises(PermissionError) as error:
            with open(f'/proc/self/fd/{fd}','r+b',buffering=0) as f:f.write(b'X')
        assert error.value.errno in (errno.EACCES,errno.EPERM) and os.pread(fd,len(orig),0)==orig


@pytest.mark.parametrize('i',range(4))
def test_changed_origin_rejected_even_on_resume(case,i):
    args,files,worker,_=case
    def stop(n):raise InterruptedError
    with pytest.raises(InterruptedError):replay.generate(*args,runner=worker,committed=stop)
    b=files[i].read_bytes(); files[i].write_bytes(bytes([b[0]^1])+b[1:]);
    with pytest.raises(ValueError,match='changed'):replay.generate(*args,runner=worker)
    assert not args[5].exists()


@pytest.mark.parametrize('kind',['corrupt','truncated','foreign','busy','plan'])
def test_dirty_checkpoint_or_competing_lease(case,kind):
    args,_,worker,d=case;c=args[4]
    def stop(n):raise InterruptedError
    with pytest.raises(InterruptedError):replay.generate(*args,runner=worker,committed=stop)
    if kind=='busy':
        with replay.sr.writer(c),pytest.raises(ValueError,match='another slab writer'):replay.generate(*args,runner=worker)
    else:
        if kind=='plan':
            d['budget_mib']=4;args[0].write_bytes(replay.sr.encode(d));args[1]=sha(args[0].read_bytes())
        else:
            with closing(sqlite3.connect(c/'slabs.sqlite3')) as db,db:
                db.execute({'corrupt':"UPDATE slabs SET payload=zeroblob(36)",'truncated':"UPDATE slabs SET payload=x'01'",'foreign':"INSERT INTO slabs VALUES(999,x'01')"}[kind])
        with pytest.raises(ValueError):replay.generate(*args,runner=worker)
    assert not args[5].exists()


@pytest.mark.parametrize('kind',['empty-dir','existing-dir','dangling-link','same-file'])
def test_noreplace_atomic_namespace(case,kind):
    args,_,worker,_=case;dest=args[-1]
    def competing(n):
        if n!=2:return
        if kind=='dangling-link':dest.symlink_to('absent')
        elif kind=='same-file':dest.write_bytes(b'sentinel')
        else:
            dest.mkdir()
            if kind=='existing-dir':(dest/'keep').write_bytes(b'sentinel')
    with pytest.raises(FileExistsError):replay.generate(*args,runner=worker,committed=competing)
    if kind=='dangling-link':assert os.readlink(dest)=='absent'
    elif kind=='same-file':assert dest.read_bytes()==b'sentinel'
    elif kind=='existing-dir':assert (dest/'keep').read_bytes()==b'sentinel'
    else:assert not list(dest.iterdir())


def test_race_two_publishers_keep_one(tmp_path):
    dirs=[tmp_path/'a',tmp_path/'b']; dest=tmp_path/'out'
    for d in dirs:d.mkdir();(d/'owner').write_text(d.name)
    from concurrent.futures import ThreadPoolExecutor
    from threading import Barrier
    start=Barrier(2)
    def race(d):
        start.wait(timeout=2)
        try:publish_new(d,dest);return True
        except FileExistsError:return False
    with ThreadPoolExecutor(max_workers=2) as pool:winners=list(pool.map(race,dirs))
    assert winners.count(True)==1 and (dest/'owner').read_text()==dirs[winners.index(True)].name
    assert (dirs[winners.index(False)]/'owner').is_file()


@pytest.mark.parametrize('kind',['reversed','extra','short','wrong-origin','false-guard'])
def test_bad_candidate_cannot_commit(case,kind):
    args,_,worker,_=case
    def bad(fds,out,d,s):
        stats=worker(fds,out,d,s);b=out.read_bytes()
        if kind=='reversed':out.write_bytes(b[::-1])
        if kind=='extra':out.write_bytes(b+b'x')
        if kind=='short':out.write_bytes(b[:-1])
        if kind=='wrong-origin':stats['roi_origin_xyz'][-1]+=1
        if kind=='false-guard':stats['redzones_verified']=False
        return stats
    with pytest.raises(ValueError):replay.generate(*args,runner=bad)
    assert not args[-1].exists()
    with closing(sqlite3.connect(args[4]/'slabs.sqlite3')) as db:assert db.execute('SELECT count(*) FROM slabs').fetchone()==(0,)


def test_private_workspace_and_no_alias(case):
    args,_,worker,_=case; args[-1]=args[4]/'nested'
    with pytest.raises((ValueError,FileNotFoundError)):replay.generate(*args,runner=worker)
    assert not args[4].exists()
