"""Evaluate the experimental calibrated candidate; reference masks for the test region are opened only after predictions."""
import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from denoise import Denoise
from local_model import median_stack
from pore_check import ROOT, native, reference
from residual_model import features, fit, probability
from review import write_review
from validate import ORIGIN, SPEC, final_gates, objects, score, sha, stack, verified


def run(i,cache,out,fetch,variant="A"):
    from block_model import features as patch_features, train_threshold, HALO
    is_b=variant=="B"; extract=patch_features if is_b else features
    if not __debug__: raise RuntimeError('Acceptance cannot run with -O')
    cache.mkdir(parents=True,exist_ok=True);out.mkdir(parents=True,exist_ok=True);stem=f'sample{i}'
    raw,rs=verified(cache,i,'raw',fetch);seg,ss=verified(cache,i,'segmented',fetch);space=(SPEC[i][2],)*3
    def blocks(a,b,gpu):
        # Preserve actual source halos across both ends of every processing batch.
        for start in range(a,b,4):
            stop=min(start+4,b);yield start,stop,extract(median_stack(raw,i,start,stop,HALO if is_b else 18),i,gpu)
    with Denoise(ROOT/'build/pores'/('patches.ptx' if is_b else 'nlm.ptx'),patch=is_b) as gpu:
        train=[(x,m,stack(seg,i,'segmented',a,b)) for a,b,(x,m) in blocks(160,184,gpu)]
        if is_b:
            model,t=train_threshold(train)
        else:
            model=fit(train[:4]); truth=np.concatenate([t for x,m,t in train[4:]]);cal=np.concatenate([probability(model,x,m) for x,m,t in train[4:]])
            # Tie-break toward 0.5 then the lower threshold.
            _,_,_,t=max((score(cal>=v,truth)['dice'],-abs(v-.5),-v,v) for v in [j/40 for j in range(6,35)])
            model=fit(train);del truth,cal
        del train
        prob=np.concatenate([probability(model,x,m) for a,b,(x,m) in blocks(400,528,gpu)])
    pred=prob>=t;table=out/(stem+'-pores.csv');spacing=space
    with TemporaryDirectory(dir=cache) as temp:
        d=Path(temp);src=d/'score.f32';(-prob).astype('<f4').tofile(src)
        st=native(src,table,pred.shape,spacing,-t,6,1000,'none',4);lab,rows=reference(pred,6,spacing,table,st)
        second=native(src,d/'b.csv',pred.shape,spacing,-t,6,37,'none',64)
        if (st['slab_depth'],second['slab_depth'])!=(8,37) or (d/'b.csv').read_bytes()!=table.read_bytes():raise ValueError('Slab invariance failed')
    ref=stack(seg,i,'segmented',400,528);m=score(pred,ref);o=objects(lab,ref,rows,out/(stem+'-matches.csv'));g=final_gates(m,o)
    viewer=out/(stem+'-review.html');audit=write_review(viewer,i,stack(raw,i,'raw',400,528),ref,lab,rows,ORIGIN,space[0]);report=dict(specimen=i,role='development' if i in [2,3,5] else 'confirmation',method=('blockwise NLM classifier B' if is_b else 'calibrated NLM residual classifier A'),
        source={'raw':rs,'segmented':ss},calibration_z=[160,184],evaluation_z=[400,528],origin_xyz=ORIGIN,probability_threshold=t,
        metrics=m,objects=o,final_gates=g,final_passed=all(g.values()),sensitivity={str(q):score(prob>=q,ref) for q in [max(0,t-.05),min(1,t+.05)]},
        table_sha256=sha(table),review_sha256=sha(viewer),review=audit,same_mask_scipy_rows=len(rows),second_slab_depth=second['slab_depth'],native=st)
    (out/(stem+'.json')).write_text(json.dumps(report,indent=2)+'\n',encoding='utf8');print(stem,report['final_passed'],m,o,flush=True);return report['final_passed']

if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--cache',type=Path,default=ROOT/'data/pores');ap.add_argument('--model',choices=['A','B'],default='A');ap.add_argument('--out',type=Path);ap.add_argument('--fetch',action='store_true');ap.add_argument('--specimens',type=int,nargs='+',choices=list(SPEC),required=True);args=ap.parse_args()
    results=[run(i,args.cache,args.out or ROOT/('build/candidate-'+args.model),args.fetch,args.model) for i in args.specimens];raise SystemExit(0 if all(results) else 1)
