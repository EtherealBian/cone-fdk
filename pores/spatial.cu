// Valid NCHW convolution for the calibrated plane network. A block writes one
// output channel; lanes at neighboring X positions read coalesced image samples.
extern "C" __global__ void spatial_conv(const float *src,const float *weights,const float *bias,
  float *dst,int width,int height,int cin,int cout,int k,int dilation,int relu){
    int ow=width-(k-1)*dilation, oh=height-(k-1)*dilation;
    int x=blockIdx.x*blockDim.x+threadIdx.x, y=blockIdx.y*blockDim.y+threadIdx.y;
    if(x>=ow || y>=oh)return;
    int z=blockIdx.z/cout, o=blockIdx.z%cout;
    float sum=bias[o];
    for(int c=0;c<cin;++c){
        for(int j=0;j<k;++j)for(int i=0;i<k;++i){
            int pos=((z*cin+c)*height+y+j*dilation)*width+x+i*dilation;
            sum=fmaf(src[pos],weights[((o*cin+c)*k+j)*k+i],sum);
        }
    }
    dst[((z*cout+o)*oh+y)*ow+x]=relu?fmaxf(sum,0.f):sum;
}
