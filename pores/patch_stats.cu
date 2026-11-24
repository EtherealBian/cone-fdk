// ROI input is the integer-valued uint16 median carried in FP32. For those inputs
// both the mean-ratio and variance-ratio tests use exact integer arithmetic.
extern "C" __global__ void nlm(const float* src, float* out, int w, int h, float bw,
                               int screen, float unused) {
    __shared__ float tile[40][40];
    __shared__ unsigned sums[36][36];
    __shared__ unsigned long long dispersions[36][36];
    int tx=threadIdx.x, ty=threadIdx.y, lane=ty*16+tx;
    int x0=16*blockIdx.x, y0=16*blockIdx.y, z=blockIdx.z;
    for(int k=lane;k<1600;k+=256) {
        int x=min(max(x0+k%40-12,0),w-1), y=min(max(y0+k/40-12,0),h-1);
        tile[k/40][k%40]=src[(z*h+y)*w+x];
    }
    __syncthreads();
    for(int k=lane;k<1296;k+=256) {
        int x=k%36, y=k/36; unsigned s=0; unsigned long long sq=0;
        for(int j=0;j<5;j++) for(int i=0;i<5;i++) {
            unsigned a=(unsigned)tile[y+j][x+i]; s+=a; sq+=(unsigned long long)a*a;
        }
        sums[y][x]=s; dispersions[y][x]=25ull*sq-(unsigned long long)s*s;
    }
    __syncthreads();
    if(x0+tx>=w || y0+ty>=h) return;
    unsigned a=sums[ty+10][tx+10]; auto b=dispersions[ty+10][tx+10];
    double num=0., den=0.; float inv=1.f/(25.f*bw*bw);
    for(int j=0;j<21;j++) for(int i=0;i<21;i++) {
        unsigned c=sums[ty+j][tx+i]; auto v=dispersions[ty+j][tx+i];
        if(screen && (20u*min(a,c)<19u*max(a,c) || 2ull*min(b,v)<max(b,v))) continue;
        float d=0.f;
        #pragma unroll
        for(int u=0;u<5;u++) {
            #pragma unroll
            for(int k=0;k<5;k++) {
                float delta=tile[ty+10+u][tx+10+k]-tile[ty+j+u][tx+i+k];
                d=fmaf(delta,delta,d);
            }
        }
        float wt=__expf(-d*inv); num+=(double)wt*tile[ty+j+2][tx+i+2]; den+=wt;
    }
    out[(z*h+y0+ty)*w+x0+tx]=num/den;
}
