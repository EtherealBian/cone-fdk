// Plane is packed as [mapped NLM intensity, unconstrained correction]. The second
// output half returns the quantized latent image to make the decision inspectable.
extern "C" __global__ void nlm(const float* src,float* dst,int width2,int height,float contrast,int,float){
    int w=width2/2, x0=16*blockIdx.x, y0=16*blockIdx.y, z=blockIdx.z;
    if(x0>=w)return; // uniform block branch before the barrier
    __shared__ unsigned char u[26][26];
    int tx=threadIdx.x,ty=threadIdx.y;
    for(int k=16*ty+tx;k<676;k+=256){
        int xx=min(max(x0+k%26-5,0),w-1), yy=min(max(y0+k/26-5,0),height-1);
        int p=(z*height+yy)*width2+xx;
        float v=src[p]+8.f*tanhf(src[p+w]*0.125f);
        u[k/26][k%26]=(unsigned char)fminf(255.f,fmaxf(0.f,floorf(v+0.5f)));
    }
    __syncthreads(); int x=x0+tx,y=y0+ty;
    if(x>=w||y>=height)return;
    int lo=255,hi=0;
    for(int j=-5;j<=5;j++)for(int i=-5;i<=5;i++)if(i*i+j*j<=26){
        int v=u[ty+5+j][tx+5+i]; lo=min(v,lo); hi=max(v,hi);
    }
    int mid=(hi+lo)/2, center=u[ty+5][tx+5]; int p=(z*height+y)*width2+x;
    dst[p]=(hi-lo<(int)contrast)?127.5f-mid:mid-center-.5f; dst[p+w]=center;
}
