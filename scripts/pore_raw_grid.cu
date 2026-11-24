// Native-grid interpolation of three axial raw fields, box-decimated at scanner phase zero.
// Input origin and halo are multiples of s. Output includes the eight-pixel offset halo.
extern "C" __global__ void raw_grid(const unsigned short* a,float* out,int w,int h,int s,double gain,double shift){
    int x=blockIdx.x*blockDim.x+threadIdx.x,y=blockIdx.y*blockDim.y+threadIdx.y;
    int ow=w-16, oh=h-16;
    if(x>=ow||y>=oh)return;
    double gx=(x+8.5)/s-0.5,gy=(y+8.5)/s-0.5;
    int bx=(int)floor(gx),by=(int)floor(gy); double dx=gx-bx,dy=gy-by;
    double v[3]={0,0,0}; int plane=w*h;
    for(int j=0;j<2;++j)for(int i=0;i<2;++i){
        double f=(j?dy:1-dy)*(i?dx:1-dx); if(f==0)continue;
        int c=0, pair=0, diff=0;
        for(int t=0;t<s;++t)for(int u=0;u<s;++u){int k=((by+j)*s+t)*w+(bx+i)*s+u;
            int lo=a[k],hi=a[2*plane+k]; c+=a[plane+k]; pair+=lo+hi; diff+=abs(lo-hi);}
        v[0]+=f*c; v[1]+=f*0.5*pair; v[2]+=f*diff;
    }
    int k=y*ow+x; double den=s*s;
    for(int c=0;c<3;++c)out[c*ow*oh+k]=(float)((((v[c]/den)+(c==2?0:shift))*gain-(c==2?0:128))/48);
}
