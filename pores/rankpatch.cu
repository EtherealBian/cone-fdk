// Rank-adaptive patch estimates. Each equal-distance bin is only partially used
// at the cutoff; the result never depends on the order of neighbours within a bin.
extern "C" __global__ void rankpatch(const float* src, float* out, int w, int h, float bandwidth, int, float) {
    __shared__ float tile[40][40];
    const int tx=threadIdx.x, ty=threadIdx.y, x=blockIdx.x*16+tx, y=blockIdx.y*16+ty, z=blockIdx.z;
    for(int k=ty*16+tx;k<1600;k+=256){
        int u=max(0,min(w-1,(int)blockIdx.x*16+k%40-12));
        int v=max(0,min(h-1,(int)blockIdx.y*16+k/40-12));
        tile[k/40][k%40]=src[(z*h+v)*w+u];
    }
    __syncthreads(); if(x>=w || y>=h) return;
    const float inv=1.f/(25.f*bandwidth*bandwidth);
    int count[12]={}; float sum[12]={}, correction[12]={};
    for(int j=-10;j<=10;j++) for(int i=-10;i<=10;i++) {
        float d2=0, mu=0;
        #pragma unroll
        for(int v=-2;v<=2;v++){
            #pragma unroll
            for(int u=-2;u<=2;u++){
                float d=tile[ty+12+v][tx+12+u]-tile[ty+12+j+v][tx+12+i+u];
                d2=fmaf(d,d,d2); mu+=d;
            }
        }
        // Exact powers of two: bins [0,1/32),[1/32,1/16),...,[32,inf).
        float q=d2*inv;
        int b=q<.03125f ? 0 : min(11,6 + int(__float_as_uint(q) >> 23) - 127);
        count[b]++; sum[b]+=tile[ty+12+j][tx+12+i]; correction[b]+=mu*.04f;
    }
    const int ranks[3]={8,32,96};
    #pragma unroll
    for(int r=0;r<3;r++){
        int remain=ranks[r]; float s=0,c=0;
        for(int b=0;b<12 && remain;b++){
            int n=min(remain,count[b]);
            if(n){float f=(float)n/count[b]; s=fmaf(f,sum[b],s); c=fmaf(f,correction[b],c); remain-=n;}
        }
        int dst=(z*6*h+y)*w+x;
        out[dst+r*h*w]=s/ranks[r];
        out[dst+(r+3)*h*w]=(s+c)/ranks[r];
    }
}
