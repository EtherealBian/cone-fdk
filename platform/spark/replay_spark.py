"""Verified crash-resumable replay on local Spark storage. Shared immutable-portable receipt format."""
import argparse
import hashlib
import json
import os
import sqlite3
import stat
import subprocess
import sys
import tempfile
from contextlib import ExitStack, closing, contextmanager
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import slab_replay as sr
from sealed_inputs import publish_new, snapshot


@contextmanager
def sealed_scan(scan, build, plan):
    geometry, projections, exe, ptx = sr.paths(scan, build)
    with ExitStack() as stack:
        pins = [plan['geometry_sha256'], plan['projections_sha256'], *(plan['products'][n] for n in sr.NAMES)]
        limits = [1 << 20, 1 << 30, 64 << 20, 64 << 20]
        fds = tuple(stack.enter_context(snapshot(p, pin, limit=limit, exact=plan['projections_bytes']
                        if i == 1 else None, executable=i == 2))
                    for i,(p,pin,limit) in enumerate(zip([geometry,projections,exe,ptx],pins,limits,strict=True)))
        yield fds


def native(fds, out, d, slab):
    geom, projections, exe, ptx = [f'/proc/self/fd/{n}' for n in fds]
    n, z, depth = d['n'], slab['z'], slab['depth']
    r = subprocess.run(list(map(str, [exe,'--ptx',ptx,'--geom',geom,'--input',projections,
        '--output',out,'--n',n,'--fov',d['fov_mm'],'--batch',d['batch'],'--budget',d['budget_mib'],
        '--roi',f'0,0,{z},{n},{n},{depth}'])), pass_fds=fds, capture_output=True, text=True,timeout=1200)
    if r.returncode:
        raise RuntimeError(f'CUDA worker exit {r.returncode}: {r.stderr[-1500:]}')
    return json.loads(r.stdout)


def private_location(path):
    p = Path(os.path.abspath(path))
    sr.need(p == p.resolve(strict=False), 'private output must not traverse symlinks')
    parent = p.parent.stat()
    sr.need(stat.S_ISDIR(parent.st_mode) and parent.st_uid == os.getuid() and parent.st_mode & 0o077 == 0,
            'cache and destination require an owner-only parent directory')
    return p


def validate_cache(cache):
    if not cache.exists():
        cache.mkdir(mode=0o700)
        with (cache/'format.txt').open('xb') as f:
            f.write(sr.MARKER); f.flush(); os.fsync(f.fileno())
    sr.need(cache.is_dir() and cache.stat().st_uid==os.getuid() and cache.stat().st_mode & 0o077==0,'not a private slab cache')
    for p in cache.iterdir():
        s = p.lstat()
        sr.need(not p.is_symlink() and s.st_uid==os.getuid(), 'unexpected checkpoint link/owner')
        if p.name.startswith('.compute-'):
            sr.need(stat.S_ISDIR(s.st_mode), 'invalid abandoned temporary directory')
        else:
            sr.need(p.name in {'format.txt','writer.sqlite3','writer.sqlite3-journal','slabs.sqlite3','slabs.sqlite3-journal'}
                    and stat.S_ISREG(s.st_mode) and s.st_nlink == 1,'unexpected checkpoint entry')
    sr.need((cache/'format.txt').is_file() and sr.bounded(cache/'format.txt',128)==sr.MARKER,'not a private slab cache')


def generate(plan_file, pin, scan, build, cache, dest, *, committed=None, runner=native):
    d = sr.plan(plan_file, pin)
    cache, dest = private_location(cache), private_location(dest)
    sr.need(not os.path.lexists(dest) and not cache.is_relative_to(dest) and not dest.is_relative_to(cache)
            and all(not p.resolve().is_relative_to(s) and not s.is_relative_to(p.resolve())
                    for p in (plan_file,*sr.paths(scan,build)) for s in (cache,dest)), 'fresh separate output required')
    reused = reused_bytes = 0; stats = []
    with sealed_scan(scan, build, d) as fds:
        validate_cache(cache)
        with sr.writer(cache), closing(sqlite3.connect(cache/'slabs.sqlite3', timeout=0)) as db:
            saved = sr.attach(db,pin,{s['z']:sr.size(d,s) for s in d['slabs']})
            for slab in d['slabs']:
                z, length = slab['z'], sr.size(d,slab)
                if z in saved:
                    with db.blobopen('slabs','payload',z,readonly=True) as blob:
                        sr.transfer(blob,length,slab['sha256'])
                    reused += 1; reused_bytes += length; continue
                with tempfile.TemporaryDirectory(dir=cache,prefix='.compute-') as tmp:
                    raw = Path(tmp)/'slab.f32'; report = runner(fds,raw,d,slab); sr.attest(report,d,slab)
                    with db, raw.open('rb') as src:
                        db.execute('INSERT INTO slabs VALUES(?,zeroblob(?))',(z,length))
                        with db.blobopen('slabs','payload',z) as blob:
                            sr.transfer(src,length,slab['sha256'],blob)
                        sr.need(not src.read(1),'trailing native slab')
                stats.append(dict(z=z,depth=slab['depth'],gpu=report['gpu'],driver=report['driver_api'],
                                  peak_owned_bytes=report['peak_owned_bytes']))
                if committed: committed(len(stats))
            with tempfile.TemporaryDirectory(dir=dest.parent,prefix='.assemble-') as td:
                stage = Path(td)/'delivery'; stage.mkdir(mode=0o700); h=hashlib.sha256()
                with (stage/'volume.f32').open('xb') as out:
                    for s in d['slabs']:
                        with db.blobopen('slabs','payload',s['z'],readonly=True) as blob:
                            sr.transfer(blob,sr.size(d,s),s['sha256'],dest=out,all_bytes=h)
                    sr.need(h.hexdigest()==d['volume_sha256'],'assembled volume SHA-256 mismatch')
                    out.flush(); os.fsync(out.fileno())
                content=sr.encode(sr.manifest(d,pin))
                with (stage/'manifest.json').open('xb') as f:
                    f.write(content); f.flush(); os.fsync(f.fileno())
                publish_new(stage,dest)
    return dict(reused=reused,computed=len(stats),reused_bytes=reused_bytes,native=stats,
                volume_sha256=d['volume_sha256'],manifest_sha256=hashlib.sha256(content).hexdigest(),
                native_projection_bytes_per_invocation=d['projections_bytes'])


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['plan_file','scan','build','cache','dest']: p.add_argument(name,type=Path)
    p.add_argument('--plan-pin',required=True)
    a=p.parse_args(); print(json.dumps(generate(a.plan_file,a.plan_pin,a.scan,a.build,a.cache,a.dest)))
