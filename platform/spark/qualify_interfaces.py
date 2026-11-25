"""Run the unchanged interface qualification with an independently reproduced ARM64 build."""
import argparse
import json
import sys
from pathlib import Path

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];sys.path.insert(0,str(ROOT/'scripts'))
from anchor_audit import publish
from build_periodic_labels import sha
from qualify_periodic_interfaces import BUILD_NAMES,SCALE,qualify
from qualify_periodic_labels import need,stress

HISTORICAL='3d2316032de01bb07b871759102bdcbf9aa89bd6ed27ddad8a953d909b059a8b'


def run(first,second,work):
    old=ROOT/'reports/interface-qualification.json'; need(sha(old)==HISTORICAL,'historical reference changed')
    work.mkdir(mode=0o700,exist_ok=False); build=work/'build';build.mkdir(); p=first/'build.json';q=second/'build.json'
    need(first.stat().st_ino!=second.stat().st_ino and p.read_bytes()==q.read_bytes(),'independent build receipts differ')
    rec=json.loads(p.read_bytes());need(rec['machine']=='aarch64' and rec['arch']=='compute_121','not the Spark build')
    for name,v in rec['products'].items():
        a,b=first/name,second/name; need(not a.samefile(b) and sha(a)==sha(b)==v,'different or aliased builds')
    pins={}
    for alias in sorted(BUILD_NAMES):
        name=alias.removesuffix('.exe');(build/alias).symlink_to((first/name).resolve());pins[alias]=rec['products'][name]
    publish(build/'build.json',dict(format='CFDK-periodic-interface-build-v1',platform='DGX Spark/Linux aarch64',
            identical_builds=2,compiler=rec['compiler'],cuda=rec['nvcc'],arch=rec['arch'],products=pins,
            full_build_sha256=sha(p),sources={str(f.relative_to(ROOT)):sha(f) for f in [HERE/'build.py',
            HERE/'files_posix.hpp',HERE/'lib_linux.hpp',HERE/'finite_portable.hpp',ROOT/'src/periodic_interfaces.cpp',
            ROOT/'src/periodic_interfaces.cu',ROOT/'src/gpu.hpp']}))
    scale=work/'scale.u8'
    with scale.open('xb') as f:
        for a in stress(1536,960): f.write(a.tobytes())
    need(sha(scale)==SCALE,'regenerated pressure mask did not match external pin')
    out=work/'cases';qualify(build,scale,out)
    result=json.loads((out/'qualification.json').read_bytes());prev=json.loads(old.read_bytes());need(result['gpu']=='NVIDIA GB10','wrong GPU')
    need(set(result['workloads'])==set(prev['workloads']) and set(result['refusals'])==set(prev['refusals']),'different qualification scope')
    for name,case in result['workloads'].items():
        for field in ('input_sha256','face_sha256','shape_ha','maps','cells','runs'):
            need(case[field]==prev['workloads'][name][field],f'{field} differs from historical result in {name}')
        for band in (1,7,32):need(sha(out/f'{name}-b{band}.u8')==case['face_sha256'],'saved plane changed')
    publish(work/'cross-platform.json',dict(format='CFDK-GB10-interface-comparison-v1',historical_sha256=HISTORICAL,
        spark_qualification_sha256=sha(out/'qualification.json'),hardware=result['gpu'],
        map_count=sum(c['maps'] for c in result['workloads'].values()),cells=sum(c['cells'] for c in result['workloads'].values()),
        band_rows=[1,7,32],workloads=len(result['workloads']),negative_controls=sorted(result['refusals']),
        scope='Same typed sampled-grid interface bytes and bounded buffer accounting. Analytic/pressure protocol, not physical dimensional truth.'))
    print((work/'cross-platform.json').read_text())


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('first','second','work'):p.add_argument(k,type=Path)
    run(**{k:v.resolve() for k,v in vars(p.parse_args()).items()})
