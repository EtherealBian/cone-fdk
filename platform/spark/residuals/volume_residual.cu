// Bounded, deterministic reduction of co-registered little-endian binary32 volumes.
// Host side is a reader only. Publication/provenance is the qualification caller's job.
#include <cuda_runtime.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <sys/stat.h>
#include <unistd.h>
#include <vector>

namespace {
constexpr unsigned Threads = 128, Tile = Threads * 8, Guard = 256;
struct Stats {
    double bias, absolute, sse, reference, maximum;
    unsigned first, count, equal, over[4], invalid;
};
__host__ __device__ Stats initial() {
    Stats r{};
    r.first = 0xffffffffu;
    return r;
}
__host__ __device__ void append(Stats& a, const Stats& b) {
    a.bias += b.bias; a.absolute += b.absolute; a.sse += b.sse; a.reference += b.reference;
    if (b.maximum > a.maximum || (b.maximum == a.maximum && b.first < a.first)) {
        a.maximum = b.maximum; a.first = b.first;
    }
    a.count += b.count; a.equal += b.equal; a.invalid += b.invalid;
    for (unsigned k=0;k<4;++k) a.over[k] += b.over[k];
}
__global__ void residual(const float* a, const float* b, Stats* output, unsigned plane) {
    __shared__ Stats partial[Threads];
    const unsigned t=threadIdx.x, base=blockIdx.y*plane, end=min(plane,(blockIdx.x+1)*Tile);
    Stats s=initial();
    for (unsigned i=blockIdx.x*Tile+t;i<end;i+=Threads) {
        const double x=double(a[base+i]), y=double(b[base+i]);
        if (!isfinite(x) || !isfinite(y)) {++s.invalid;continue;}
        const double d=x-y, v=fabs(d);
        ++s.count; s.equal += (x==y); s.bias+=d; s.absolute+=v; s.sse+=d*d; s.reference+=y*y;
        if (v>s.maximum || (v==s.maximum && i<s.first)) {s.maximum=v;s.first=i;}
        s.over[0]+=v>0x1p-20; s.over[1]+=v>0x1p-18; s.over[2]+=v>0x1p-16; s.over[3]+=v>0x1p-14;
    }
    partial[t]=s;__syncthreads();
    for (unsigned step=Threads/2;step;step/=2) {
        if (t<step) append(partial[t],partial[t+step]);
        __syncthreads();
    }
    if (!t) output[blockIdx.y*gridDim.x+blockIdx.x]=partial[0];
}
void need(bool ok,const char* why) {if(!ok) throw std::runtime_error(why);}
void ck(cudaError_t error) {if(error!=cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));}
unsigned number(const char* s,unsigned lo,unsigned hi) {
    size_t n=std::strlen(s);need(n>0 && n<=8 && std::strspn(s,"0123456789")==n,"bad decimal argument");
    auto x=std::stoul(s);need(x>=lo && x<=hi,"argument outside bounds");return unsigned(x);
}
struct Input {
    int fd;
    Input(const char* name, size_t expected):fd(open(name,O_RDONLY|O_NONBLOCK|O_CLOEXEC)) {
        need(fd>=0,"cannot open volume");struct stat s{};
        need(fstat(fd,&s)==0 && S_ISREG(s.st_mode) && size_t(s.st_size)==expected,"volume type or length mismatch");
    }
    void take(void* buf,size_t left) {
        auto* next=static_cast<unsigned char*>(buf);
        while(left) {ssize_t n=read(fd,next,left);need(n>0,"volume shortened or read failed");left-=size_t(n);next+=n;}
    }
    void finish() {char b;need(read(fd,&b,1)==0,"volume extended or final read failed");}
    ~Input(){if(fd>=0)close(fd);}
};
struct Buffer {
    unsigned char* start=nullptr;size_t bytes;
    explicit Buffer(size_t n):bytes(n){ck(cudaMalloc(&start,n+2*Guard));ck(cudaMemset(start,0xa5,Guard));ck(cudaMemset(start+Guard+n,0x5a,Guard));}
    template<typename T> T* data(){return reinterpret_cast<T*>(start+Guard);}
    void check() const {
        unsigned char v[Guard];
        for(unsigned end=0;end<2;++end) {
            ck(cudaMemcpy(v,start+(end?Guard+bytes:0),Guard,cudaMemcpyDeviceToHost));
            for(auto b:v) need(b==(end?0x5a:0xa5),"device redzone changed");
        }
    }
    ~Buffer(){if(start)cudaFree(start);}
};
struct Event {
    cudaEvent_t e;Event(){ck(cudaEventCreate(&e));}~Event(){cudaEventDestroy(e);}
};
using Clock=std::chrono::steady_clock;
double milliseconds(Clock::time_point t) {return std::chrono::duration<double,std::milli>(Clock::now()-t).count();}
void run(int argc,char** argv) {
    need(argc==8,"usage: volume-residual candidate.f32 reference.f32 nx ny nz band budget_mib");
    const unsigned nx=number(argv[3],1,2048),ny=number(argv[4],1,2048),nz=number(argv[5],1,2048),requested=number(argv[6],1,64),budget=number(argv[7],1,1024);
    const unsigned plane=nx*ny, band=std::min(nz,requested), tiles=(plane+Tile-1)/Tile;
    const size_t total=size_t(plane)*nz, cap=size_t(band)*plane, outcap=size_t(band)*tiles;
    const size_t owned=2*cap*sizeof(float)+outcap*sizeof(Stats)+6*Guard;
    need(total<=134217728ULL && owned<=(size_t(budget)<<20),"voxel or device budget exceeded");
    Input a(argv[1],4*total),b(argv[2],4*total);
    auto begin=Clock::now();cudaDeviceProp p{};ck(cudaGetDeviceProperties(&p,0));ck(cudaSetDevice(0));
    Buffer x(cap*sizeof(float)),y(cap*sizeof(float)),out(outcap*sizeof(Stats));Event start,stop;
    std::vector<float> ha(cap),hb(cap);std::vector<Stats> partial(outcap),rows(nz);
    double kernel_ms=0,read_ms=0;unsigned launches=0;size_t uploaded=0,downloaded=0;
    for(unsigned z=0;z<nz;z+=band) {
        unsigned depth=std::min(band,nz-z);size_t count=size_t(plane)*depth,bytes=count*sizeof(float),dbytes=size_t(tiles)*depth*sizeof(Stats);
        auto t=Clock::now();a.take(ha.data(),bytes);b.take(hb.data(),bytes);read_ms+=milliseconds(t);
        ck(cudaMemcpy(x.data<float>(),ha.data(),bytes,cudaMemcpyHostToDevice));ck(cudaMemcpy(y.data<float>(),hb.data(),bytes,cudaMemcpyHostToDevice));uploaded+=2*bytes;
        ck(cudaEventRecord(start.e));residual<<<dim3(tiles,depth),Threads>>>(x.data<float>(),y.data<float>(),out.data<Stats>(),plane);ck(cudaGetLastError());ck(cudaEventRecord(stop.e));ck(cudaEventSynchronize(stop.e));
        float ms=0;ck(cudaEventElapsedTime(&ms,start.e,stop.e));kernel_ms+=ms;++launches;
        ck(cudaMemcpy(partial.data(),out.data<Stats>(),dbytes,cudaMemcpyDeviceToHost));downloaded+=dbytes;
        for(unsigned k=0;k<depth;++k) {
            Stats s=initial();for(unsigned i=0;i<tiles;++i) append(s,partial[k*tiles+i]);
            need(!s.invalid && s.count==plane && s.first<plane,"nonfinite voxel or incomplete plane");rows[z+k]=s;
        }
    }
    a.finish();b.finish();x.check();y.check();out.check();const double wall_ms=milliseconds(begin);
    // Print only after the final plane, EOF, finite checks and guard checks succeed.
    std::cout << std::setprecision(17) << "{\"format\":\"CFDK-GB10-volume-residual-v1\",\"shape_xyz\":["<<nx<<","<<ny<<","<<nz<<"],\"gpu\":\""<<p.name<<"\",\"band\":"<<band<<",\"tile_size\":"<<Tile<<",\"partial_bytes\":"<<sizeof(Stats)<<",\"owned_bytes\":"<<owned<<",\"budget_bytes\":"<<(size_t(budget)<<20)<<",\"host_payload_bytes\":"<<2*cap*sizeof(float)+outcap*sizeof(Stats)+nz*sizeof(Stats)<<",\"h2d_bytes\":"<<uploaded<<",\"d2h_bytes\":"<<downloaded<<",\"launches\":"<<launches<<",\"guard_checks\":6,\"read_ms\":"<<read_ms<<",\"kernel_ms\":"<<kernel_ms<<",\"wall_ms\":"<<wall_ms<<",\"rows\":[";
    for(unsigned z=0;z<nz;++z) {const auto& s=rows[z];if(z)std::cout<<",";
        std::cout<<"["<<s.count<<","<<s.equal;for(auto v:s.over)std::cout<<","<<v;
        std::cout<<","<<s.maximum<<","<<s.first<<","<<s.bias<<","<<s.absolute<<","<<s.sse<<","<<s.reference<<"]";}
    std::cout<<"]}\n";
}
}
int main(int argc,char** argv) {try {run(argc,argv);return 0;}catch(const std::exception& e){std::cerr<<"volume-residual: "<<e.what()<<"\n";return 2;}}
