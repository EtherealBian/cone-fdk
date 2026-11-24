// Seven monotone distance-weight bases, stored as separate sums and masses.
// Input floats are exact uint16 samples; accumulate patch SSD in integer arithmetic.
extern "C" __global__ void nlm(const float* src,float* out,int w,int h,float raw_per_unit,int,float){
    __shared__ unsigned short tile[40][40];
    int tx=threadIdx.x,ty=threadIdx.y,bx=16*blockIdx.x,by=16*blockIdx.y,z=blockIdx.z;
    for(int k=16*ty+tx;k<1600;k+=256){int x=min(max(bx+k%40-12,0),w-1),y=min(max(by+k/40-12,0),h-1);tile[k/40][k%40]=(unsigned short)src[(z*h+y)*w+x];}
    __syncthreads();if(bx+tx>=w||by+ty>=h)return;
    const double unit2=25.*raw_per_unit*raw_per_unit;
    const int radii[6]={8,16,24,32,48,64};
    double num[7]={},den[7]={};
    for(int j=-10;j<=10;j++)for(int i=-10;i<=10;i++){
        unsigned long long s=0;
        #pragma unroll
        for(int v=-2;v<=2;v++){
            #pragma unroll
            for(int u=-2;u<=2;u++){int delta=(int)tile[ty+12+v][tx+12+u]-(int)tile[ty+12+j+v][tx+12+i+u];unsigned d=abs(delta);s+=(unsigned long long)d*d;}
        }
        int px=tile[ty+12+j][tx+12+i];double wt=exp(-(double)s/(900.*unit2));num[0]+=wt*px;den[0]+=wt;
        #pragma unroll
        for(int b=0;b<6;b++)if((double)s<=radii[b]*radii[b]*unit2){num[b+1]+=px;den[b+1]+=1.;}
    }
    #pragma unroll
    for(int b=0;b<7;b++){int p=((z*14+b)*h+by+ty)*w+bx+tx;out[p]=float(num[b]);out[p+7*h*w]=float(den[b]);}
}
