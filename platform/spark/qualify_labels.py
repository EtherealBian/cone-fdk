"""Use the unchanged label qualification and six AB/BA end-to-end trials on GB10."""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from anchor_audit import publish
from build_periodic_labels import sha
from qualify_periodic_labels import checked, need

HIST = '494033edb46586219f85f3f80835629429a7fb32f8df86b7616c7322f00628cb'


def run(first, second, sources, frozen, work):
    historical=checked(ROOT/'reports/periodic-label-qualification.json',HIST)
    work.mkdir(mode=0o700,exist_ok=False)
    out=work/'build'; out.mkdir()
    rec=json.loads((first/'build.json').read_bytes())
    need(rec['machine']=='aarch64' and rec['arch']=='compute_121' and (first/'build.json').read_bytes()==(second/'build.json').read_bytes(),'independent Spark build identities')
    for name,pin in rec['products'].items():
        a,b=first/name,second/name
        need(not a.samefile(b) and sha(a)==sha(b)==pin,'different or aliased product '+name)
    products={}
    for name in ('periodic-labels.exe','periodic-labels.ptx'):
        n=name.removesuffix('.exe'); (out/name).symlink_to((first/n).resolve());products[name]=rec['products'][n]
    upstream={n:sha(frozen/('scripts' if n.endswith('.py') else 'src')/n) for n in historical['build']['sources']}
    need(upstream==historical['build']['sources'],'exact R8 native inputs required')
    need(all(sha(frozen/'scripts'/n)==v for n,v in historical['verification_sources'].items()), 'exact R8 verification sources required')
    for n in ('periodic_labels.cpp','periodic_labels.cu','gpu.hpp'):
        need(sha(ROOT/'src'/n)==upstream[n], 'active native algorithm differs from R8')
    need(all(sha(first/'src'/n)==sha(second/'src'/n) for n in ('periodic_labels.cpp','periodic_labels.cu','gpu.hpp','files.hpp')), 'generated native sources differ')
    need(sha(first/'src/files.hpp')==sha(HERE/'files_posix.hpp'), 'native filesystem adapter differs')
    publish(out/'build.json',dict(format='CFDK-periodic-label-build-v1', platform='Linux ARM64/GNU + CUDA',
        identical_builds=2, compiler=rec['compiler'],cuda=rec['nvcc'],arch=rec['arch'],products=products,
        sources=upstream, actual_compiled_sources={n:sha(first/'src'/n) for n in ('periodic_labels.cpp','periodic_labels.cu','files.hpp','gpu.hpp')},
        port_sha256={n:sha(HERE/n) for n in ('build.py','lib_linux.hpp','files_posix.hpp')}, full_build_sha256=sha(first/'build.json')))
    cases=work/'cases'
    # Run with ROOT bound naturally by the historical script's real location. No pin changes.
    env=dict(os.environ); env.pop('PYTHONPATH',None);env['PYTHONDONTWRITEBYTECODE']='1'
    subprocess.run([sys.executable, frozen/'scripts/qualify_periodic_labels.py',out,ROOT/'reports',sources/'bridge',sources/'legacy',cases],cwd=work,env=env,check=True)
    r=json.loads((cases/'qualification.json').read_bytes());need(r['gpu']=='NVIDIA GB10','wrong GPU')
    for key in ('protocol_sha256','workloads','source_references','verification_sources'):
        need(r[key]==historical[key], 'non-timing historical evidence differs: '+key)
    need(set(r['refusals'])==set(historical['refusals']),'refusal coverage')
    publish(work/'comparison.json',dict(format='CFDK-Spark-label-comparison-v1',
        qualification_sha256=sha(cases/'qualification.json'), historical_sha256=HIST,
        harness_sha256=sha(Path(__file__)), build_manifest_sha256=sha(out/'build.json'),
        gpu=r['gpu'],maps=sum(c['maps'] for c in r['workloads'].values()),
        cells=sum(c['pair_cells'] for c in r['workloads'].values()), performance_workload=r['workloads']['scale'],
        timing=r['timing'],boundary=r['timing_boundary'],scope='Sampled ternary masks; comparison is to CPU on this host, not to another platform or physical defect truth.'))
    print((work/'comparison.json').read_text())


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('first','second','sources','frozen','work'):p.add_argument(n,type=Path)
    run(**{k:v.absolute() for k,v in vars(p.parse_args()).items()})
