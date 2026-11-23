"""Blockwise estimates of pore boundaries. All spatial filters operate on real scan halos."""
import numpy as np
from scipy import ndimage as ndi

from local_model import SETTINGS
from residual_model import fit, probability

HALO = 24

def aggregate(patches, radius):
    """Overlap-add predictions: patch offset (dy,dx) estimates pixel (y+dy,x+dx)."""
    a = np.zeros_like(patches[:, 0]); n = 0
    for dy in range(-radius, radius+1):
        for dx in range(-radius, radius+1):
            a += np.roll(patches[:, 5*dy+dx+12], (dy,dx), axis=(1,2)); n += 1
    # The caller crops HALO; wrapped pixels never participate in a retained feature.
    return a/n

def features(median, specimen, gpu):
    slope,offset,ct=SETTINGS[specimen]; h=HALO; arrays=[]; keep=None
    crop=lambda a:a[:,h:-h,h:-h]
    yy,xx=np.ogrid[-5:6,-5:6];foot=((xx*xx+yy*yy)<=26)[None]
    for band,noise,weighted in ((14,0,0),(22,0,0),(30,0,0),(42,0,0),(22,15,1)):
        patches=gpu(median,band/slope,weighted,noise/slope)
        for r in (0,2):
            image=(aggregate(patches,r)+offset)*slope
            low=crop(ndi.minimum_filter(image,footprint=foot));high=crop(ndi.maximum_filter(image,footprint=foot))
            local=crop(image);low=np.clip(low,-.5,255.49);high=np.clip(high,-.5,255.49)
            mid=(high+low)*.5; contrast=high-low; margin=mid-local
            eligible=((margin>=-11)&(contrast>=ct*.25))|(mid<149)
            keep=eligible if keep is None else keep|eligible
            arrays.extend([local,mid,contrast,margin])
            if r==2:
                for sigma in (.8,1.7):
                    avg=ndi.gaussian_filter(image,(0,sigma,sigma),truncate=3)
                    arrays.append(crop(avg)-local)
                # Smaller neighbourhoods disambiguate steep gradients and weak small pores.
                av=ndi.uniform_filter(image,(1,3,3)); arrays.append(crop(av)-local)
        del patches
    arrays.append(crop(median).astype('f4')*slope)
    return np.stack([a[keep] for a in arrays],axis=1).astype('f4'),keep

def train_threshold(blocks):
    # Six consecutive 4-slice blocks: predict 8 slices per fold. Exclude a
    # neighbouring 4-slice guard block, so shared 3D-median inputs cannot straddle the split.
    probs=[]; labels=[]
    for k in (0,2,4):
        subset=[b for j,b in enumerate(blocks) if j<k-1 or j>k+2];model=fit(subset)
        for x,mask,truth in blocks[k:k+2]:
            probs.append(probability(model,x,mask));labels.append(truth)
    p=np.concatenate(probs).ravel(); truth=np.concatenate(labels).ravel(); n=int(truth.sum())
    # Minimize a combination of Dice loss and absolute *relative* count bias on
    # out-of-fold calibration predictions. Includes unclassified (0-probability) voxels.
    v=np.arange(15,86,dtype=float)/100; positive=np.sort(p[truth]); negative=np.sort(p[~truth]);tp=n-np.searchsorted(positive,v);fp=len(negative)-np.searchsorted(negative,v)
    loss=1-2*tp/(tp+fp+n)+.4*np.abs(tp+fp-n)/max(1,n)
    return fit(blocks),float(v[np.argmin(loss)])
