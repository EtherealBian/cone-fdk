// Three packed integer fields: admitted donor votes, original face degree, foreign within L1<=2.
extern "C" __global__ void shell_fields(const unsigned char* owner,const unsigned char* donors,
        unsigned char* packed,int nz,int ny,int nx) {
    int k=blockIdx.x*blockDim.x+threadIdx.x;
    if(k>=nx*ny*nz)return;
    int x=k%nx,y=(k/nx)%ny,z=k/(nx*ny),deg=0,foreign=0;
    for(int dz=-2;dz<=2;dz++) for(int dy=-2;dy<=2;dy++) for(int dx=-2;dx<=2;dx++) {
        int d=abs(dx)+abs(dy)+abs(dz),a=z+dz,b=y+dy,c=x+dx;
        if(d>2 || a<0 || a>=nz || b<0 || b>=ny || c<0 || c>=nx)continue;
        int q=owner[(a*ny+b)*nx+c]; deg+=d==1 && q==1; foreign|=q==2;
    }
    packed[k]=__popc((unsigned)donors[k])|(deg<<3)|(foreign<<6);
}
