"""Reproduce the five-slab qualification on Spark, without altering historical inputs or gates."""
import argparse
from contextlib import closing
import json
import os
from pathlib import Path
import sqlite3
import sys

from replay_spark import HERE, ROOT, generate, sr
import qualify_slab_replay as old

PLAN = ROOT/'fixtures/walnut512-slab-replay-spark-v1.json'
PIN = 'bb25b41dae5f7fdccdc58c6692313addafca96973a44c99cc1b97ceec15bd047'


def interrupt(scan, build, cache, dest):
    def killed(n):
        if n==2: os._exit(73)
    generate(PLAN,PIN,scan,build,cache,dest,committed=killed)
    raise RuntimeError('worker failed to interrupt')


def qualify(scan, build, dense, out):
    d=sr.plan(PLAN,PIN); legacy=sr.plan(old.PLAN,old.PLAN_PIN)
    sr.need({k:v for k,v in d.items() if k!='products'}=={k:v for k,v in legacy.items() if k!='products'},
            'changed measurement inputs, settings, partition or expected output')
    sr.need(old.sha(ROOT/'fixtures/slab-replay-protocol-v1.json')==old.RULES,'legacy protocol changed')
    reports=ROOT/'reports'
    for name,pin in old.REFERENCE.items(): sr.need(old.sha(reports/name)==pin,'historical report changed')
    source=json.loads((reports/'native-ab.json').read_bytes())['cases']['full-corrected']
    baseline=json.loads((reports/'run-repeat.json').read_bytes())['baseline_outputs']['walnut512']
    sr.need(source['geom_sha256']==d['geometry_sha256'] and source['projections_sha256']==d['projections_sha256']
            and baseline==d['volume_sha256']==old.sha(dense),'baseline acquisition changed')
    out.mkdir(mode=0o700,exist_ok=False); cache=out/'resume-cache'; resumed=out/'resumed'; cold=out/'cold'; coldcache=out/'cold-cache'
    code=old.child([Path(__file__),'_interrupt',scan,build,cache,resumed],expected=73)
    sr.need(not os.path.lexists(resumed),'premature delivery')
    with closing(sqlite3.connect(f'file:{cache}/slabs.sqlite3?mode=ro', uri=True)) as db:
        checkpoint=db.execute('SELECT z,length(payload) FROM slabs ORDER BY z').fetchall()
    sr.need(checkpoint==[(s['z'],sr.size(d,s)) for s in d['slabs'][:2]],'wrong committed prefix')
    results={}
    for n,c,dest,start in [('resumed',cache,resumed,2),('cold',coldcache,cold,0)]:
        r=old.child([HERE/'replay_spark.py',PLAN,scan,build,c,dest,'--plan-pin',PIN]);results[n]=r
        sr.need((r['reused'],r['computed'],r['reused_bytes'])==(start,5-start,sum(sr.size(d,s) for s in d['slabs'][:start])),'wrong scheduled counts')
        sr.need([m['z'] for m in r['native']]==[s['z'] for s in d['slabs'][start:]] and all(m['gpu']=='NVIDIA GB10' for m in r['native']),'wrong GPU or slab jobs')
        sr.need(r['volume_sha256']==baseline and r['manifest_sha256']==old.sha(dest/'manifest.json'),'unexpected output')
        sr.need(old.child(['-I','-S',old.WORKER,'verify',PLAN,dest,'--plan-pin',PIN])==sr.manifest(d,PIN),'offline verification failed')
    sr.need(results['resumed']['manifest_sha256']==results['cold']['manifest_sha256'],'manifest mismatch')
    slabs=old.compare_outputs(dense,[resumed,cold],d)
    q=dict(format='CFDK-spark-slab-replay-qualification-v1',scope='720 measured views; deterministic process crash/recovery only, not physical dimensional accuracy or power-loss testing',
           plan_sha256=PIN,legacy_plan_sha256=old.PLAN_PIN,legacy_protocol_sha256=old.RULES,historical_reports=old.REFERENCE,
           original_volume_sha256=baseline,directly_compared_voxels=d['n']**3,products=d['products'],
           source_sha256={str(p.relative_to(ROOT)):old.sha(p) for p in [Path(__file__),HERE/'replay_spark.py',HERE/'sealed_inputs.py',old.WORKER,Path(old.__file__)]},
           interrupted={'exit_code':code,'complete_output_present':False,'committed_z_and_bytes':checkpoint},
           checkpoint_bytes={key:(p/'slabs.sqlite3').stat().st_size for key,p in [('resumed',cache),('cold',coldcache)]},slabs_compared=slabs,**results)
    with (out/'qualification.json').open('xb') as f: f.write(sr.encode(q));f.flush();os.fsync(f.fileno())
    print(json.dumps(dict(quit=code,checkpoint=checkpoint,resumed=[results['resumed']['reused'],results['resumed']['computed']],cold=[results['cold']['reused'],results['cold']['computed']],direct_voxels=q['directly_compared_voxels'],sha256=baseline)))


if __name__=='__main__':
    if sys.argv[1:2]==['_interrupt']: interrupt(*map(Path,sys.argv[2:]))
    else:
        a=argparse.ArgumentParser(description=__doc__)
        for k in ['scan','build','dense','out']:a.add_argument(k,type=Path)
        qualify(**vars(a.parse_args()))
