"""Layer isolation: measure the published mask directly; this is NOT an automatic segmentation score."""
import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from pore_check import ROOT, native, reference
from validate import ORIGIN, SPEC, sha, stack, verified


def run(i,cache,out):
    if not __debug__: raise RuntimeError('Validation needs assertions')
    out.mkdir(exist_ok=True,parents=True);path,source=verified(cache,i,'segmented',False)
    mask=stack(path,i,'segmented',400,528);spacing=(SPEC[i][2],)*3;table=out/f'sample{i}-reference-pores.csv'
    with TemporaryDirectory(dir=cache) as temp:
        d=Path(temp);src=d/'in.f32';(~mask).astype('<f4').tofile(src)
        st=native(src,table,mask.shape,spacing,0,6,1000,'none',4);_,rows=reference(mask,6,spacing,table,st)
        other=native(src,d/'check.csv',mask.shape,spacing,0,6,37,'none',64)
        if st['slab_depth']==other['slab_depth'] or table.read_bytes()!=(d/'check.csv').read_bytes():raise ValueError('Partition invariance failure')
    report=dict(input='Published NIST binary mask, NOT raw-to-mask detection',specimen=i,source=source,origin_xyz=ORIGIN,
        count_error_voxels=st['pore_voxels']-int(mask.sum()),all_rows_checked=len(rows),float_rtol=1e-12,
        slab_depths=[st['slab_depth'],other['slab_depth']],table_sha256=sha(table),native=st)
    (out/f'sample{i}-reference.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf8');print(i,len(rows),report['count_error_voxels'])

if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--cache',type=Path,default=ROOT/'data/pores');ap.add_argument('--out',type=Path,default=ROOT/'build/mask-oracle');ap.add_argument('--specimens',type=int,nargs='+',default=[2,3,5],choices=list(SPEC));args=ap.parse_args()
    for specimen in args.specimens:run(specimen,args.cache,args.out)
