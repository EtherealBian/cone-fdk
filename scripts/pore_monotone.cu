// One residue class per launch. Two candidate centers in a class are >=5 apart.
// Neither reads the other's four-voxel neighborhood; default stream orders classes.
extern "C" __global__ void coordinate(const unsigned char* base, unsigned char* cur,
    const unsigned short* raw,const int* w,long long* gains,int W,int H,int Z,
    int pore2,int material2,long long norm,int color){
    int nx=(W+4)/5,ny=(H+4)/5,nz=(Z+4)/5;
    int t=blockIdx.x*blockDim.x+threadIdx.x;if(t>=nx*ny*nz)return;
    int x=5*(t%nx)+color%5,y=5*((t/nx)%ny)+(color/5)%5,z=5*(t/(nx*ny))+color/25;
    if(x<4||y<4||z<4||x>=W-4||y>=H-4||z>=Z-4)return;
    const int i=(z*H+y)*W+x,plane=H*W,delta=material2-pore2;
    int adj[6]={1,-1,W,-W,plane,-plane};int edges=0,n=0;
    for(int s:adj){edges+=(base[i]!=base[i+s]);n+=cur[i+s];}
    if(!edges)return;
    long long a=0;
    for(int dz=-2;dz<=2;++dz)for(int dy=-2;dy<=2;++dy)for(int dx=-2;dx<=2;++dx){
        int shell=dz*dz+dy*dy+dx*dx;if(shell>4 || !w[shell])continue;
        int j=i+(dz*H+dy)*W+dx,field=0;
        for(int c=-2;c<=2;++c)for(int b=-2;b<=2;++b)for(int e=-2;e<=2;++e){
            int s=c*c+b*b+e*e;if(s<=4)field+=w[s]*cur[j+(c*H+b)*W+e];
        }
        a+=(long long)w[shell]*(98304LL*(material2-2*(int)raw[j])-(long long)delta*field);
    }
    int m=cur[i];long long vote=12*a+(long long)delta*norm*(9*(2*m-1)+n-3);
    long long gain=m?-vote:vote;
    if(gain>0){gains[i]=gain;cur[i]=1-m;}
}
