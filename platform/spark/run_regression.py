"""Spark-only, fresh-directory regression runner; keep receipts and failing per-test scratch."""
import argparse
import collections
import fcntl
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

from build_publishers import ROOT, build

PRESETS={'historical':['tests'], 'spark':['platform/spark/tests'],
         'periodic':['tests/test_periodic_parcel.py','tests/test_periodic_interfaces.py','tests/test_periodic_roi.py']}


def run(job, preset):
    if sys.platform!='linux' or platform.machine()!='aarch64':raise ValueError('native Spark only')
    m=ROOT.parent; job=job.absolute(); parent=job.parent
    if job!=job.resolve(strict=False) or not parent.is_dir() or parent.stat().st_uid!=os.getuid() or parent.stat().st_mode&0o077:
        raise ValueError('new job below private real directory required')
    overlay=ROOT/'build/spark/python'; pin=build(overlay); job.mkdir(mode=0o700,exist_ok=False)
    lib=m/'vendor/astra-install/lib'; env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',PYTHONPATH=f'{overlay}:{ROOT}/scripts',
        LD_LIBRARY_PATH=str(lib)+(os.pathsep+os.environ['LD_LIBRARY_PATH'] if os.environ.get('LD_LIBRARY_PATH') else ''))
    if not (lib/'libastra.so.0').is_file(): raise ValueError('qualified private ASTRA library unavailable')
    xml=job/'result.xml';log=job/'pytest.log';args=[sys.executable,'-B','-m','pytest','-q','-p','no:cacheprovider',
        '-o','pythonpath=build/spark/python scripts','-o','tmp_path_retention_policy=failed',*PRESETS[preset],
        f'--basetemp={job}/tmp',f'--junitxml={xml}']
    with (m/'gpu.lock').open('rb') as lock,log.open('xb') as out:
        fcntl.flock(lock,fcntl.LOCK_EX); start=time.monotonic();result=subprocess.run(args,cwd=ROOT,env=env,stdout=out,stderr=subprocess.STDOUT)
        elapsed=time.monotonic()-start
    counts=collections.Counter()
    if xml.is_file():
        for case in ET.parse(xml).iter('testcase'):
            counts[next((k for k in ['failure','error','skipped'] if case.find(k) is not None),'passed')]+=1
    if result.returncode==0 and not counts: raise ValueError('empty success report')
    hashes={f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in (log,xml) if f.is_file()}
    receipt=dict(format='CFDK-Spark-test-run-v1',preset=preset,overlay_sha256=pin,command=args,seconds=elapsed,
        process_exit=result.returncode,counts=dict(counts),files_sha256=hashes,
        retention='Pytest retains failed per-test scratch. Explicit private basetemp; shared session fixtures may remain. Full logs/XML retained.')
    (job/'run.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps({'returncode':result.returncode,'counts':dict(counts)}))
    return result.returncode


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('preset',choices=list(PRESETS));p.add_argument('job',type=Path)
    a=p.parse_args();sys.exit(run(a.job,a.preset))
