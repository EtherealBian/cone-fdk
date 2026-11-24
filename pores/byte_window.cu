// Native, integer-only byte-window dictionary inference. One output byte per voxel.
__device__ int resolve(int key, int prior, const int* keys, const int* votes, int n) {
    int left=0,right=n;
    while(left<right){int m=left+(right-left)/2;if(keys[m]<key)left=m+1;else right=m;}
    if(left==n || keys[left]!=key)return prior;
    int v=votes[left]+(prior?8:-8);
    return v==0?prior:v>0;
}
extern "C" __global__ void lookup(const int* codes, const int* rk, const int* rv,
    const int* ek, const int* ev, unsigned char* out,int n,int nr,int ne) {
    int j=blockIdx.x*blockDim.x+threadIdx.x;if(j>=n)return;
    unsigned int t=codes[j];int hi=t&255, u=(t>>8)&255,lo=(t>>16)&255,ct=t>>24;
    int mid=(lo+hi)/2,span=hi-lo,src=span<ct?mid<128:u<mid;
    int r=(((mid<128)*512+span-ct+255)*512+mid-u+255);
    int rel=resolve(r,src,rk,rv,nr); int ex=resolve(t,rel,ek,ev,ne);
    out[j]=src+2*rel+4*ex;
}
