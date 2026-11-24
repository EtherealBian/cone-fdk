// Six bandwidths for one registered geometry. Input is the native uint16 median
// represented exactly in float32. Block-local squared-difference box sums are
// exact integers in FP64 for all 11x11 uint16 patches, even at full scale.
extern "C" __global__ void adaptive(const float *src,float *dst,int w,int h,float unit,int wide,float patch){
    __shared__ float tile[68][68];
    __shared__ double squares[26][26], horizontal[26][16];
    const int tx=threadIdx.x, ty=threadIdx.y, lane=16*ty+tx;
    const int x=16*blockIdx.x+tx, y=16*blockIdx.y+ty,z=blockIdx.z;
    const int r=wide?21:10,p=patch>3?5:2, halo=r+p, edge=16+2*halo, side=16+2*p;
    for(int q=lane;q<edge*edge;q+=256){int yy=q/edge,xx=q%edge;
        int sy=min(h-1,max(0,int(16*blockIdx.y)+yy-halo)), sx=min(w-1,max(0,int(16*blockIdx.x)+xx-halo)); tile[yy][xx]=src[(z*h+sy)*w+sx];}
    __syncthreads(); double mass[6]={}, signal[6]={}; float inv[6];
    for(int b=0;b<6;b++){float bw=(15.f+5.f*b)*unit;inv[b]=1.f/((2*p+1)*(2*p+1)*bw*bw);}
    for(int dy=-r;dy<=r;dy++)for(int dx=-r;dx<=r;dx++){
        for(int q=lane;q<side*side;q+=256){int yy=q/side,xx=q%side; double d=tile[r+yy][r+xx]-tile[r+yy+dy][r+xx+dx]; squares[yy][xx]=d*d;}
        __syncthreads();
        for(int j=ty;j<side;j+=16){double v=0.;for(int k=0;k<=2*p;k++)v+=squares[j][tx+k];horizontal[j][tx]=v;}
        __syncthreads(); double d=0.;for(int k=0;k<=2*p;k++)d+=horizontal[ty+k][tx];
        float val=tile[halo+ty+dy][halo+tx+dx];
        #pragma unroll
        for(int b=0;b<6;b++){double a=__expf(-float(d)*inv[b]);mass[b]+=a;signal[b]+=a*val;}
        __syncthreads();
    }
    if(x<w && y<h)for(int b=0;b<6;b++)dst[((z*6+b)*h+y)*w+x]=float(signal[b]/mass[b]);
}
