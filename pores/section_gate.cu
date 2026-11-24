// Each z-plane is a separate XY object crop: labels must never cross batch planes.
// Codes: core 0, exposed edge pixels 1..4 (number of open edges), exterior 5/6, unused 7.
extern "C" __global__ void xy_rings(const unsigned char* m,unsigned char* out,int nz,int ny,int nx){
    int k=blockIdx.x*blockDim.x+threadIdx.x;if(k>=nx*ny*nz)return;
    int y=(k/nx)%ny,x=k%nx,deg=0,dist=4;
    if(m[k]==2){out[k]=7;return;}
    for(int dy=-3;dy<=3;++dy)for(int dx=-3;dx<=3;++dx){
        int d=abs(dx)+abs(dy),x1=x+dx,y1=y+dy;
        if(d>3 || x1<0 || y1<0 || x1>=nx || y1>=ny)continue;
        if(m[k+dy*nx+dx]==1){deg+=(d==1);dist=min(dist,d);}
    }
    out[k]=m[k]==1 ? 4-deg : dist==1 ? 5 : dist<=3 ? 6 : 7;
}
