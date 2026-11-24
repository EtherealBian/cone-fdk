// Rank four of the real center and its six face neighbors. Input has halo 4.
extern "C" __global__ void seed_median7(const unsigned short* raw, unsigned short* out,
                                         int w, int h, int d) {
    const int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=w*h*d) return;
    const int pitch=w+8, plane=pitch*(h+8);
    const int k=(i/(w*h)+4)*plane+((i/w)%h+4)*pitch+i%w+4;
    unsigned short v[7]={raw[k],raw[k-1],raw[k+1],raw[k-pitch],raw[k+pitch],raw[k-plane],raw[k+plane]};
    #pragma unroll
    for(int j=1;j<7;++j){unsigned short t=v[j]; int a=j; while(a && v[a-1]>t){v[a]=v[a-1];--a;} v[a]=t;}
    out[i]=v[3];
}
