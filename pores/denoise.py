"""Small checked Windows CUDA Driver launcher for the experimental NLM kernel."""
import ctypes as c
from pathlib import Path

import numpy as np


class Denoise:
    def __init__(self,ptx,patch=False):
        self.patch=patch
        self.d=c.WinDLL('nvcuda'); self.ctx=c.c_void_p(); self.module=c.c_void_p(); self.kernel=c.c_void_p(); self.dev=c.c_int()
        self.calls('cuInit',0);self.calls('cuDeviceGet',c.byref(self.dev),0)
        self.calls('cuDevicePrimaryCtxRetain',c.byref(self.ctx),self.dev)
        self.calls('cuCtxSetCurrent',self.ctx); self.calls('cuModuleLoadData',c.byref(self.module),Path(ptx).read_bytes());self.calls('cuModuleGetFunction',c.byref(self.kernel),self.module,b'patches' if patch else b'nlm')
    def calls(self,name,*args):
        rc=getattr(self.d,name)(*args)
        if rc: raise RuntimeError(f'{name} failed: {rc}')
    def close(self):
        self.calls('cuModuleUnload',self.module); self.calls('cuDevicePrimaryCtxRelease_v2',self.dev)
    def __enter__(self):return self
    def __exit__(self,*exc):self.close()
    def __call__(self,a,h,weighted=0,noise=0.):
        if not np.isfinite(h) or h<=0: raise ValueError('Invalid bandwidth')
        a=np.ascontiguousarray(a,dtype=np.float32);nz,ny,nx=a.shape;out=np.empty((nz,25,ny,nx),np.float32) if self.patch else np.empty_like(a)
        # Short dispatches avoid exceeding the WDDM display-driver timeout.
        cap=min(nz,2); size=cap*ny*nx*4; p,q=c.c_uint64(),c.c_uint64(); allocated=[]
        try:
            for ptr in (p,q):self.calls('cuMemAlloc_v2',c.byref(ptr),c.c_size_t(size*(25 if self.patch and ptr is q else 1)));allocated.append(ptr)
            args=[p,q,c.c_int(nx),c.c_int(ny),c.c_float(h),c.c_int(weighted),c.c_float(noise)];argv=(c.c_void_p*len(args))(*(c.addressof(x) for x in args))
            for z in range(0,nz,cap):
                aa=a[z:z+cap];bb=out[z:z+cap];self.calls('cuMemcpyHtoD_v2',p,c.c_void_p(aa.ctypes.data),c.c_size_t(aa.nbytes));self.calls('cuLaunchKernel',self.kernel,(nx+15)//16,(ny+15)//16,len(aa),16,16,1,0,None,argv,None);self.calls('cuMemcpyDtoH_v2',c.c_void_p(bb.ctypes.data),q,c.c_size_t(bb.nbytes))
        finally:
            for ptr in allocated:self.calls('cuMemFree_v2',ptr)
        return out
