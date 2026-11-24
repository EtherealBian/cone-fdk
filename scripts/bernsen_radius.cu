// Classify a strict Bernsen bit at the given center and noncenter extrema.
static __device__ bool pore(int c, int l, int h, int ct) {
    l = min(l,c); h = max(h,c);
    return h-l < ct ? l+h < 256 : l+h >= 2*c+2;
}
// For fixed neighboring bytes the pore indicator is nonincreasing in center
// (ct>=2). Outside the neighbor range a high-contrast extreme is necessarily
// pore at the dark end and solid at the bright end. Inside, the midpoint is fixed.
static __device__ bool reachable(bool target, int e, int center, int nlo, int nhi, int ct) {
    int c = target ? max(0,center-e) : min(255,center+e);
    int h1 = min(255,nhi+e);
    for(int l=max(0,nlo-e); l<=min(255,nlo+e); ++l){
        int m=min(c,l), first=max(c,max(l,max(0,nhi-e))), last=max(c,h1);
        // The input extrema have equal-radius intervals and their original owners
        // may realize the new extrema. Other neighbors can be clamped between them.
        if(first-m<ct){
            int end=min(last,m+ct-1);
            if(target ? m+first<256 : m+end>=256) return true;
        }
        if(last-m>=ct){
            int start=max(first,m+ct);
            if(target ? m+last>=2*c+2 : m+start<2*c+2) return true;
        }
    }
    return false;
}
// Reuse the checked planar CUDA launcher. Inputs and output signed codes are
// integers exactly representable as float; thresholds are the published integers.
extern "C" __global__ void nlm(const float *u, float *out, int w, int h, float contrast, int, float){
    int x=blockIdx.x*blockDim.x+threadIdx.x, y=blockIdx.y*blockDim.y+threadIdx.y;
    if(x>=w||y>=h) return;
    int k=blockIdx.z*w*h, c=int(u[k+y*w+x]), lo=255,hi=0,ct=int(contrast);
    for(int j=-5;j<=5;++j)for(int i=-5;i<=5;++i){
        if((i==0&&j==0)||i*i+j*j>26) continue;
        int v=int(u[k+min(h-1,max(0,y+j))*w+min(w-1,max(0,x+i))]);lo=min(lo,v);hi=max(hi,v);
    }
    bool p=pore(c,lo,hi,ct);
    int a=0,b=9;
    while(b-a>1){int m=(a+b)/2;if(reachable(!p,m,c,lo,hi,ct))b=m;else a=m;}
    out[k+y*w+x]=float(p ? b : -b);
}
