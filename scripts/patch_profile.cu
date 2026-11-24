// Pixelwise NLM, changing only the spatial coefficients of the patch distance.
// Source uint16 values are represented exactly; weighted integer SSD is exact
// in double for both coefficient sums (25 or 256). Shared tile has a real halo.
extern "C" __global__ void adaptive(const float* src, float* dst, int w, int h,
                                    float unit, int binomial, float /*unused*/) {
    __shared__ float tile[40][40];
    __shared__ double square[20][20], horizontal[20][16];
    int tx=threadIdx.x, ty=threadIdx.y, lane=16*ty+tx;
    int x=16*blockIdx.x+tx, y=16*blockIdx.y+ty, z=blockIdx.z;
    const int coeff[5]={1,4,6,4,1};
    for(int q=lane;q<1600;q+=256){
        int i=q/40, j=q%40;
        int yy=min(h-1,max(0,int(16*blockIdx.y)+i-12));
        int xx=min(w-1,max(0,int(16*blockIdx.x)+j-12));
        tile[i][j]=src[(z*h+yy)*w+xx];
    }
    __syncthreads();
    double mass[6]={}, signal[6]={}; float inv[6];
    for(int b=0;b<6;++b){float bw=(15.f+5.f*b)*unit; inv[b]=1.f/((binomial?256.f:25.f)*bw*bw);}
    for(int dy=-10;dy<=10;++dy)for(int dx=-10;dx<=10;++dx){
        for(int q=lane;q<400;q+=256){int i=q/20,j=q%20;
            double d=tile[10+i][10+j]-tile[10+i+dy][10+j+dx]; square[i][j]=d*d;}
        __syncthreads();
        for(int i=ty;i<20;i+=16){double sum=0.;for(int j=0;j<5;++j)sum+=(binomial?coeff[j]:1)*square[i][tx+j];horizontal[i][tx]=sum;}
        __syncthreads();
        double s=0.;for(int i=0;i<5;++i)s+=(binomial?coeff[i]:1)*horizontal[ty+i][tx];
        float value=tile[12+ty+dy][12+tx+dx];
        #pragma unroll
        for(int b=0;b<6;++b){double a=__expf(-float(s)*inv[b]);mass[b]+=a;signal[b]+=a*value;}
        __syncthreads();
    }
    if(x<w && y<h)for(int b=0;b<6;++b)dst[((z*6+b)*h+y)*w+x]=float(signal[b]/mass[b]);
}
