"""Research candidate: scan-calibrated residual classifier. Train only on the registered calibration Z range."""
import numpy as np
from scipy import ndimage as ndi
from sklearn.ensemble import HistGradientBoostingClassifier
from threadpoolctl import threadpool_limits

from local_model import SETTINGS, XY_HALO, local_fields


def features(median,specimen,gpu):
    # Normalize bandwidth to the published scale of each scan. Add local statistics
    # rather than spatial coordinates or specimen identity; no test labels are used.
    slope,offset,ct=SETTINGS[specimen]; arrays=[]; keep=None;h=XY_HALO
    crop=lambda a:a[:,h:-h,h:-h]
    for nominal,weighted in [(14,0),(23,0),(31,0),(42,0),(30,1)]:
        den=gpu(median,nominal/slope,weighted); u,mid,contrast=local_fields(den,specimen)
        local=crop((den+offset)*slope); margin=mid-local
        k=((margin>=-8)&(contrast>=ct*.3))|(mid<145)
        keep=k if keep is None else keep|k
        arrays.extend([local,mid,contrast,margin])
        for size in [3,7]:
            av=ndi.uniform_filter(den,size=(1,size,size));arrays.append(crop((den-av)*slope))
        if nominal in (23,31):
            # Approximate the effect of a spatially regularized patch estimate.
            for sigma in (.6,1.2):
                smooth=ndi.gaussian_filter(den,sigma=(0,sigma,sigma));us,ms,cs=local_fields(smooth,specimen)
                arrays.extend([ms-us.astype('i2'),cs])
    arrays.append(crop(median).astype('f4')*slope)
    return np.stack([a[keep] for a in arrays],axis=1).astype('f4'),keep

def fit(features_and_masks,seed=173):
    xs=[];ys=[];rng=np.random.default_rng(seed)
    for x,mask,truth in features_and_masks:
        y=truth[mask];pos=np.flatnonzero(y);neg=np.flatnonzero(~y);sel=np.r_[pos,rng.choice(neg,size=min(len(neg),max(2500,20*len(pos))),replace=False)];xs.append(x[sel]);ys.append(y[sel])
    X=np.concatenate(xs);Y=np.concatenate(ys);n=len(Y)
    if Y.all() or not Y.any():raise ValueError('Calibration contains only one eligible class')
    model=HistGradientBoostingClassifier(max_iter=120,max_leaf_nodes=23,min_samples_leaf=25,l2_regularization=1.,early_stopping=False,random_state=seed)
    with threadpool_limits(4):model.fit(X,Y)
    return model

def probability(model,x,mask):
    p=np.zeros(mask.shape,'f4')
    if len(x):
        with threadpool_limits(4):p[mask]=model.predict_proba(x)[:,1]
    return p
