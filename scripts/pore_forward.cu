// Integer shell numerators: all K weights use denominator 24 * 4096.
extern "C" __global__ void blur_occ(const unsigned char* m,const int* weights,int* field,
    int w,int h,int d) {
    int i=blockDim.x*blockIdx.x+threadIdx.x; if(i>=w*h*d)return;
    int x=i%w,y=(i/w)%h,z=i/(w*h),acc=0;
    for(int dz=-2;dz<=2;dz++)for(int dy=-2;dy<=2;dy++)for(int dx=-2;dx<=2;dx++){
        int s=dz*dz+dy*dy+dx*dx; if(s>4 || weights[s]==0)continue;
        int j=(max(0,min(d-1,z+dz))*h+max(0,min(h-1,y+dy)))*w+max(0,min(w-1,x+dx));
        acc+=weights[s]*m[j];
    }
    field[i]=acc;
}
extern "C" __global__ void inverse_step(const unsigned char* m,const int* weights,const int* field,
    const unsigned short* raw,unsigned char* result,int w,int h,int d,int material2,int contrast,long long norm) {
    int i=blockDim.x*blockIdx.x+threadIdx.x; if(i>=w*h*d)return;
    int x=i%w,y=(i/w)%h,z=i/(w*h);int orig=m[i];
    if(x<4 || y<4 || z<4 || x>=w-4 || y>=h-4 || z>=d-4){result[i]=orig;return;}
    long long u=0, fitted=0;
    for(int dz=-2;dz<=2;dz++)for(int dy=-2;dy<=2;dy++)for(int dx=-2;dx<=2;dx++){
        int s=dz*dz+dy*dy+dx*dx;if(s>4 || weights[s]==0)continue;
        int j=i+(dz*h+dy)*w+dx;
        u+=(long long)weights[s]*(material2-2*(int)raw[j]);
        fitted+=(long long)weights[s]*field[j];
    }
    int n=m[i+1]+m[i-1]+m[i+w]+m[i-w]+m[i+w*h]+m[i-w*h];
    long long vote=12*(98304LL*u-(long long)contrast*fitted)+(long long)contrast*norm*(9*(2*orig-1)+n-3);
    result[i]=vote==0?orig:vote>0;
}
