// Identical 5x5 patch metric; only the set of admissible search offsets changes.
extern "C" __global__ void nlm(const float* src,float* out,int w,int h,float bw,int disk,float) {
    __shared__ float tile[40][40];
    int tx=threadIdx.x, ty=threadIdx.y, bx=blockIdx.x*16, by=blockIdx.y*16, z=blockIdx.z;
    for(int k=ty*16+tx;k<1600;k+=256){
        int x=min(max(bx+k%40-12,0),w-1),y=min(max(by+k/40-12,0),h-1);
        tile[k/40][k%40]=src[(z*h+y)*w+x];
    }
    __syncthreads(); if(bx+tx>=w || by+ty>=h) return;
    float inv=1.f/(25.f*bw*bw); double num=0.,den=0.;
    for(int j=-10;j<=10;j++)for(int i=-10;i<=10;i++){
        if(disk && i*i+j*j>100) continue;
        float d=0.f;
        #pragma unroll
        for(int v=-2;v<=2;v++){
            #pragma unroll
            for(int u=-2;u<=2;u++){
                float delta=tile[ty+12+v][tx+12+u]-tile[ty+12+j+v][tx+12+i+u];
                d=fmaf(delta,delta,d);
            }
        }
        double a=__expf(-d*inv); den+=a; num+=a*tile[ty+12+j][tx+12+i];
    }
    out[(z*h+by+ty)*w+bx+tx]=float(num/den);
}
