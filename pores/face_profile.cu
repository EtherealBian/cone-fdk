// Input positions are padded raw linear indices. Stepping outward, compare three adjacent faces.
extern "C" __global__ void face_profiles(const unsigned short* raw,const int* pos,unsigned char* out,
                                       int step,int t1,int t2,int n) {
    int k=blockIdx.x*blockDim.x+threadIdx.x; if(k>=n)return;
    int line[10],plane[10];
    for(int j=0;j<10;j++) {int p=pos[k]+(j-4)*step,s=0;line[j]=raw[p];
        for(int y=-1;y<=1;y++)for(int x=-1;x<=1;x++)s+=raw[p+y*t1+x*t2];plane[j]=s;}
    for(int j=0;j<3;j++)for(int c=0;c<2;c++) {int *v=c?plane:line;int a=v[j],b=a;
        for(int t=1;t<8;t++){a=min(a,v[j+t]);b=max(b,v[j+t]);}
        for(int t=0;t<8;t++)out[k*48+j*16+c*8+t]=(b==a?0:255*(v[j+t]-a)/(b-a));}
}
