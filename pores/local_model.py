"""Experimental publication-informed segmentation. The 18-pixel context prevents artificial ROI edges."""
import io
from zipfile import ZipFile

import numpy as np
from PIL import Image
from scipy import ndimage as ndi
from validate import SPEC

XY_HALO=18
SETTINGS={2:(.0085,-25000,15),3:(.005667,-10000,15),4:(.004857,-2500,21),5:(.006375,-20000,13),6:(.003891,0,17)}

def context_stack(path,specimen,a,b,halo=XY_HALO):
    h=halo; prefix=f'set1sample{specimen}raw'; layers=[]
    with ZipFile(path) as archive:
        for z in range(a-1,b+1):
            with Image.open(io.BytesIO(archive.read(f'{prefix}/{prefix}_{z:04d}.tif'))) as im:
                if im.size!=SPEC[specimen][:2]: raise ValueError('Unexpected scan dimensions')
                layers.append(np.array(im.crop((360-h,376-h,616+h,632+h))))
    vol=np.stack(layers);assert vol.dtype==np.uint16
    return vol

def median_stack(path,specimen,a,b,halo=XY_HALO):
    return ndi.median_filter(context_stack(path,specimen,a,b,halo),size=3,mode='nearest')[1:-1]

def local_fields(denoised,specimen,rounding=.5,rad2=26):
    slope,offset,_=SETTINGS[specimen]; u=np.clip(np.floor((denoised+offset)*slope+rounding),0,255).astype('u1'); yy,xx=np.ogrid[-5:6,-5:6]; footprint=(xx*xx+yy*yy<=rad2)[None]
    lo=ndi.minimum_filter(u,footprint=footprint);hi=ndi.maximum_filter(u,footprint=footprint)
    h=XY_HALO;sl=(slice(None),slice(h,-h),slice(h,-h));lo=lo[sl].astype('i2');hi=hi[sl].astype('i2');return u[sl],(hi+lo)//2,hi-lo

def segment(fields,contrast,bias=0):
    u,mid,diff=fields;return np.where(diff<contrast,mid<128,u.astype('i2')<mid+bias)
