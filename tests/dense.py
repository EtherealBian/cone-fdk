"""Independent FP64 oracle and repeatability: clipped detectors, both angle signs and tails."""
import pathlib
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import numpy as np

from spatial_oracle import backproject_points

with tempfile.TemporaryDirectory() as d:
    d = pathlib.Path(d)
    for a,v,u,n,sign in [(9,7,11,13,1),(19,7,11,19,-1),(17,2,2,17,1),(16,3,8,18,-1)]:
        fov=7.3
        g=SimpleNamespace(a=a,v=v,u=u,du=.67,dv=.71,R=92,D=145,start=.213,sign=sign)
        meta=d/'scan.geom'; raw=d/'scan.f32'; out=d/'out.f32'
        meta.write_text(f'CFDK1 {a} {v} {u} .67 .71 92 145 .213 {sign}\n')
        data=np.random.default_rng(2311).normal(size=(a,v,u)).astype('<f4'); data.tofile(raw)
        cmd=[sys.argv[1],str(meta),str(raw),str(n),str(fov),str(out)]
        subprocess.run(cmd,check=True); binary=out.read_bytes()
        subprocess.run(cmd,check=True); assert binary==out.read_bytes()
        z,y,x=(np.indices((n,n,n))-(n-1)/2)*fov/n
        ref,_=backproject_points(data.astype('f8'),g,np.stack((x,y,z),axis=-1)); got=np.frombuffer(binary,'<f4').reshape(ref.shape)
        assert np.linalg.norm(got-ref)/np.linalg.norm(ref)<5e-6
        assert np.max(np.abs(got-ref))/np.max(abs(ref))<1e-5
    for bad in (data.ravel()[:-1],np.full_like(data,np.nan)):
        bad.tofile(raw); assert subprocess.run(cmd,capture_output=True).returncode!=0
