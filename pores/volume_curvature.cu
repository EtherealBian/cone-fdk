// Valid 3D box sums followed by sign-symmetric directional curvatures.
extern "C" __global__ void box27(const unsigned short* raw, int* box, int z, int y, int x) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (z-2)*(y-2)*(x-2)) return;
    int p = i%(x-2) + x*(i/(x-2)%(y-2) + y*(i/((x-2)*(y-2))));
    int sum=0;
    for(int k=0;k<3;k++) for(int j=0;j<3;j++) for(int h=0;h<3;h++)
        sum += raw[p+(k*y+j)*x+h];
    box[i]=sum;
}
extern "C" __global__ void chords(const int* box,int* out,int z,int y,int x) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=(z-8)*(y-8)*(x-8)) return;
    int p=(4+i%(x-8))+x*((4+i/(x-8)%(y-8))+y*(4+i/((x-8)*(y-8))));
    for(int t=0,r=1;t<3;t++,r*=2){
        int a[13], n=0;
        for(int k=0;k<=1;k++) for(int j=-1;j<=1;j++) for(int h=-1;h<=1;h++) {
            if(k==0 && (j<0 || (j==0 && h<=0))) continue;
            int step=((k*y+j)*x+h)*r;
            int v=(box[p+step]+box[p-step]-2*box[p])*(6/(k*k+j*j+h*h));
            int m=n++; while(m>0 && a[m-1]>v){a[m]=a[m-1]; --m;} a[m]=v;
        }
        for(int j=0;j<5;j++) out[i*15+5*t+j]=a[3*j];
    }
}
