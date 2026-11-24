// Slice-wise, noise-corrected patch comparisons. Each output is a separate
// intensity estimator; the per-scan calibration chooses their contribution.
extern "C" __global__ void adaptive(const float *src,float *out,int w,int h,float bandwidth,int,float) {
    __shared__ float tile[40][40];
    int tx=threadIdx.x, ty=threadIdx.y, x=blockIdx.x*16+tx, y=blockIdx.y*16+ty, z=blockIdx.z;
    for(int k=ty*16+tx;k<1600;k+=256){
        int u=max(0,min(w-1,(int)blockIdx.x*16+k%40-12));
        int v=max(0,min(h-1,(int)blockIdx.y*16+k/40-12));
        tile[k/40][k%40]=src[(z*h+v)*w+u];
    }
    __syncthreads(); if(x>=w || y>=h) return;
    float s[6]={},ws[6]={}; float inv=1.f/(bandwidth*bandwidth);
    for(int j=-10;j<=10;j++) for(int i=-10;i<=10;i++) {
        float l2=0,l1=0,mu=0;
        #pragma unroll
        for(int v=-2;v<=2;v++) {
            #pragma unroll
            for(int u=-2;u<=2;u++) {float d=tile[ty+12+v][tx+12+u]-tile[ty+12+j+v][tx+12+i+u]; l2=fmaf(d,d,l2); l1+=fabsf(d); mu+=d;}
        }
        l2*=0.04f*inv; l1*=0.04f; mu*=0.04f;
        float dis[6]={l2*2.f, l2, fmaxf(l2-.60f,0.f)*4.f, fmaxf(l2-1.25f,0.f)*6.f,
                      l1*l1*inv*2.4f, fmaxf(l2-.75f*mu*mu*inv,0.f)*2.f+.0125f*(i*i+j*j)};
        float px=tile[ty+12+j][tx+12+i];
        #pragma unroll
        for(int n=0;n<6;n++){float wt=__expf(-dis[n]);ws[n]+=wt;s[n]=fmaf(px,wt,s[n]);}
    }
    #pragma unroll
    for(int n=0;n<6;n++)out[((z*6+n)*h+y)*w+x]=s[n]/ws[n];
}
