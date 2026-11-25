"""One measured-input Spark port gate; no model fitting or real-object accuracy claims."""
import argparse, hashlib, json, os, subprocess, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]; sys.path.insert(0,str(ROOT/'scripts'))
from common import difference
import numpy as np
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def require(ok,msg):
    if not ok:raise ValueError(msg)
p=argparse.ArgumentParser();p.add_argument('reference',type=Path);p.add_argument('out',type=Path);a=p.parse_args();ref=a.reference;out=a.out;out.mkdir(exist_ok=False,parents=True);build=ROOT/'build/spark'
plan=json.loads((ROOT/'fixtures/walnut512-slab-replay-v1.json').read_text()); inputs={name:sha(ref/name) for name in ('full-corrected.geom','full-corrected.f32','full-corrected-512.f32','full-astra-512.f32')}
for name,field in zip(list(inputs)[:3],('geometry_sha256','projections_sha256','volume_sha256')):require(inputs[name]==plan[field],name+' frozen source mismatch')
rs=[];files=[]
for i in range(2):
    f=out/f'walnut-{i}.f32';cmd=[build/'cone-fdk','--ptx',build/'kernels.ptx','--geom',ref/'full-corrected.geom','--input',ref/'full-corrected.f32','--output',f,'--n','512','--fov','40','--budget','768','--batch','16']; r=json.loads(subprocess.check_output(list(map(str,cmd)),text=True));require(f.stat().st_size==512**3*4,'size'); v=np.memmap(f,dtype='<f4',mode='r');require(np.isfinite(v).all() and np.ptp(v[256*512**2:257*512**2])>.005,'nonfinite or blank');del v;rs.append(r);files.append(f)
hashes=[sha(f) for f in files];require(hashes[0]==hashes[1],'GB10 repeat mismatch');native=difference(files[0],ref/'full-corrected-512.f32',512);astra=difference(files[0],ref/'full-astra-512.f32',512);require(native['rel_l2']<1e-4,'cross-device native rel L2');require(astra['rel_l2']<.02,'historical ASTRA rel L2');report=dict(format='cone-fdk-spark-measured-v1',comparison='historical Windows references; not physical ground truth',device_runs=rs,input_sha256=inputs,output_sha256=hashes,windows_native=native,windows_astra=astra,limits={'native_rel_l2':1e-4,'astra_rel_l2':.02},build=json.loads((build/'build.json').read_text()));(out/'result.json').write_text(json.dumps(report,indent=2)+'\n'); print(json.dumps({k:report[k] for k in ('output_sha256','windows_native','windows_astra')}))
