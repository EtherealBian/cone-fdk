"""Import a consecutive single-channel TIFF stack as a new FP32 ROI directory + provenance JSON."""
import argparse
import hashlib
import json
import math
import re
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from PIL import Image


def convert(source,dest,xy,zrange,spacing):
    if dest.exists():raise FileExistsError('Output directory must be new')
    if len(spacing)!=3 or not all(math.isfinite(s) and s>0 for s in spacing):raise ValueError('Spacing must be positive finite µm')
    x,y,w,h=xy;z0,z1=zrange
    if min(x,y,z0)<0 or min(w,h,z1-z0)<=0:raise ValueError('Invalid ROI')
    files={}
    for f in source.iterdir():
        if f.suffix.lower() not in ('.tif','.tiff'):continue
        m=re.search(r'(\d+)$',f.stem)
        if not m:raise ValueError(f'Missing numeric slice suffix: {f.name}')
        k=int(m[1]);
        if k in files:raise ValueError(f'Duplicate slice number {k}')
        files[k]=f
    dest.parent.mkdir(parents=True,exist_ok=True)
    with TemporaryDirectory(dir=dest.parent) as tmp:
        pack=Path(tmp)/'import';pack.mkdir();out=pack/'roi.f32';records=[];shape=None
        with out.open('wb') as dst:
            for k in range(z0,z1):
                if k not in files:raise ValueError(f'Missing slice {k}')
                f=files[k]
                # Hash and decode from the same read: the provenance covers the actual input bytes.
                import io
                data=f.read_bytes()
                with Image.open(io.BytesIO(data)) as im:
                    if im.n_frames!=1 or im.mode not in ('I;16','I;16B','I;16L','L','F','I'):raise ValueError(f'Unsupported TIFF: {f.name}')
                    if shape is None:shape=im.size
                    if shape!=im.size or x+w>im.width or y+h>im.height:raise ValueError('Mismatched dimensions or crop outside image')
                    a=np.asarray(im.crop((x,y,x+w,y+h)),dtype='<f4');
                    if not np.isfinite(a).all():raise ValueError('Nonfinite intensity')
                    a.tofile(dst)
                records.append(dict(index=k,file=f.name,sha256=hashlib.sha256(data).hexdigest()))
        with out.open('rb') as fp:digest=hashlib.file_digest(fp,'sha256').hexdigest()
        (pack/'volume.json').write_text(json.dumps(dict(shape_zyx=[z1-z0,h,w],origin_xyz=[x,y,z0],spacing_xyz_um=spacing,dtype='<f4',axes='ZYX/C',source_size_xy=shape,sha256=digest,slices=records),indent=2)+'\n',encoding='utf8');pack.rename(dest)

if __name__=='__main__':
    a=argparse.ArgumentParser(description=__doc__);a.add_argument('source',type=Path);a.add_argument('dest',type=Path);a.add_argument('--xywh',type=int,nargs=4,required=True);a.add_argument('--z',type=int,nargs=2,required=True,help='source slice indices [start,stop)');a.add_argument('--spacing',type=float,nargs=3,required=True,help='XYZ in µm');v=a.parse_args();convert(v.source,v.dest,v.xywh,v.z,v.spacing)
