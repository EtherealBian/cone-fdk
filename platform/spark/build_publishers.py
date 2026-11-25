"""Generate the Spark publication modules from the exact frozen source files.

Only output-name resolution, temporary staging, and publication change. Source review,
qualification, device, certificate, and detached-verification code remain upstream.
"""
import argparse
import hashlib
import json
import os
import platform
import sys
import tempfile
from pathlib import Path

from sealed_inputs import publish_new

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PINS = {
    'periodic_parcel': '71f120f18ddca655ee3f01b9fe6e2d32299a7fda3c50ace53053a59b8dea7ccd',
    'periodic_interfaces': 'fec34fd97bf47b2c2302ca4dad3c62c30f36c2d9cab4b10c574da36e17ede5a1',
    'periodic_roi': '7da3743fa06f0344eff1255d037154512a196bec5654a54a91a9e34c1f83230f',
}
IMPORT = 'import spark_publication as _spark'


def sha(data):
    return hashlib.sha256(data).hexdigest()


def one(s, old, new):
    if s.count(old) != 1:
        raise ValueError('unexpected publication boundary: ' + old)
    return s.replace(old, new)


def port(name, raw):
    if sha(raw) != PINS[name]:
        raise ValueError('frozen source SHA-256 changed: ' + name)
    s = raw.decode().replace('\r\n', '\n')
    s = one(s, 'import tempfile', IMPORT)
    if name in ('periodic_parcel', 'periodic_interfaces'):
        source = 'job' if name == 'periodic_parcel' else 'bundle'
        s = one(s, f'Path(p).resolve() for p in ({source}, dest, build, qualification)',
                f'Path({source}).resolve(), _spark.fresh(dest), Path(build).resolve(), Path(qualification).resolve()')
        s = one(s, 'tempfile.TemporaryDirectory(dir=dest.parent, prefix=', '_spark.staging(dest, prefix=')
    if name == 'periodic_parcel':
        s = one(s, '    require(\n        os.name == "nt", "CUDA parcel publisher requires Windows no-overwrite directory rename"\n    )\n', '')
        s = one(s, 'os.rename(\n            stage, dest\n        )  # Windows fails even for an already existing empty destination directory.', '_spark.publish(stage, dest)')
    if name == 'periodic_interfaces':
        s = one(s, 'os.name == "nt"\n        and not dest.exists()', 'not dest.exists()')
        s = one(s, 'fresh separate Windows output required', 'fresh separate Spark output required')
        s = one(s, 'os.rename(stage, dest)', '_spark.publish(stage, dest)')
    if name == 'periodic_roi':
        s = one(s, IMPORT, IMPORT + '\nimport tempfile')  # kept for atomic_file
        s = one(s, 'Path(bundle).resolve(), Path(dest).resolve()', 'Path(bundle).resolve(), _spark.fresh(dest)')
        s = one(s, 'atomic_file(dest, data)', '_spark.atomic_index(atomic_file, dest, data)')
        s = one(s, 'path = Path(path).resolve()\n    need(os.name == "nt", "receipt publisher requires Windows no-replace directory rename")', 'path = _spark.fresh(path)')
        s = one(s, 'tempfile.TemporaryDirectory(dir=path.parent, prefix=', '_spark.staging(path, prefix=')
        s = one(s, 'os.rename(stage, path)', '_spark.publish(stage, path)')
    compile(s, name + '.py', 'exec')
    return s.encode()


def build(out):
    if sys.platform != 'linux' or platform.machine() != 'aarch64':
        raise ValueError('Linux ARM64 qualification only')
    blobs = {name+'.py': port(name, (ROOT/'scripts'/(name+'.py')).read_bytes()) for name in PINS}
    for support in ('sealed_inputs.py', 'spark_publication.py'):
        blobs[support] = (HERE/support).read_bytes()
    manifest = dict(format='CFDK-Spark-publisher-overlay-v1', source_sha256=PINS,
                    generator_sha256=sha(Path(__file__).read_bytes()),
                    products_sha256={k: sha(v) for k,v in sorted(blobs.items())})
    blobs['overlay.json'] = (json.dumps(manifest, sort_keys=True, indent=2)+'\n').encode()
    if os.path.lexists(out):
        if (out.is_symlink() or not out.is_dir() or {p.name for p in out.iterdir()} != set(blobs)
                or any((out/n).is_symlink() or (out/n).read_bytes()!=b for n,b in blobs.items())):
            raise ValueError('existing overlay not identical; use a new directory')
    else:
        with tempfile.TemporaryDirectory(dir=out.parent) as tmp:
            stage = Path(tmp)/'overlay'; stage.mkdir()
            for n,b in blobs.items():
                (stage/n).write_bytes(b)
            publish_new(stage,out)
    return sha(blobs['overlay.json'])


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, default=ROOT/'build/spark/python')
    print(build(p.parse_args().out.absolute()))
