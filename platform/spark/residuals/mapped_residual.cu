// GB10 streaming transport. Reuse the *unchanged* protocol-v1 kernel and reduction order.
// We do not size allocations from the reported free VRAM: CPU and GPU share the pool.
#define main frozen_volume_residual_entry_point
#include "volume_residual.cu"
#undef main
#include <cerrno>
#include <memory>

namespace spark_mapped {
struct Pinned {
    unsigned char* host=nullptr;
    void* device=nullptr;
    const size_t bytes;
    explicit Pinned(size_t n):bytes(n) {
        ck(cudaHostAlloc(&host,n+2*Guard,cudaHostAllocMapped));
        auto rc=cudaHostGetDevicePointer(&device,host,0);
        if(rc!=cudaSuccess) {cudaFreeHost(host);host=nullptr;ck(rc);}
        std::memset(host,0xa5,Guard);std::memset(host+Guard+n,0x5a,Guard);
    }
    Pinned(const Pinned&)=delete;
    Pinned& operator=(const Pinned&)=delete;
    ~Pinned(){if(host)cudaFreeHost(host);}
    template<class T>T* cpu(){return reinterpret_cast<T*>(host+Guard);}
    template<class T>T* gpu(){return reinterpret_cast<T*>(static_cast<unsigned char*>(device)+Guard);}
    void check() const {
        for(unsigned i=0;i<Guard;i++)need(host[i]==0xa5 && host[Guard+bytes+i]==0x5a,"mapped redzone changed");
    }
};
void run_mapped(int argc,char** argv) {
    need(argc==8,"usage: mapped-residual candidate reference nx ny nz band application_mib");
    const unsigned nx=number(argv[3],1,2048),ny=number(argv[4],1,2048),nz=number(argv[5],1,2048),band=std::min(nz,number(argv[6],1,64)),budget=number(argv[7],1,1024);
    const unsigned plane=nx*ny, tiles=(plane+Tile-1)/Tile;
    const size_t total=size_t(plane)*nz,cap=size_t(band)*plane,outcap=size_t(band)*tiles;
    const size_t mapped=2*cap*sizeof(float)+outcap*sizeof(Stats)+6*Guard, all=mapped+nz*sizeof(Stats);
    need(total<=134217728ULL && all<=(size_t(budget)<<20),"voxel or combined payload budget exceeded");
    Input a(argv[1],4*total),b(argv[2],4*total);auto begin=Clock::now();
    ck(cudaSetDevice(0));cudaDeviceProp p{};ck(cudaGetDeviceProperties(&p,0));
    need(std::strcmp(p.name,"NVIDIA GB10")==0 && p.major==12 && p.minor==1 && p.canMapHostMemory && p.unifiedAddressing,"qualified GB10 mapped host memory required");
    int directpt=0,pageable=0,hostreg=0;ck(cudaDeviceGetAttribute(&pageable,cudaDevAttrPageableMemoryAccess,0));ck(cudaDeviceGetAttribute(&directpt,cudaDevAttrPageableMemoryAccessUsesHostPageTables,0));ck(cudaDeviceGetAttribute(&hostreg,cudaDevAttrHostRegisterSupported,0));
    Pinned x(cap*sizeof(float)),y(cap*sizeof(float)),partials(outcap*sizeof(Stats));Event start,stop;std::vector<Stats> rows(nz);
    double kernel_ms=0,read_ms=0;unsigned launches=0;size_t bytes_read=0;
    for(unsigned z=0;z<nz;z+=band) {
        const unsigned depth=std::min(band,nz-z);const size_t bytes=size_t(plane)*depth*sizeof(float);auto t=Clock::now();
        a.take(x.cpu<float>(),bytes);b.take(y.cpu<float>(),bytes);bytes_read+=2*bytes;read_ms+=milliseconds(t);
        ck(cudaEventRecord(start.e));residual<<<dim3(tiles,depth),Threads>>>(x.gpu<float>(),y.gpu<float>(),partials.gpu<Stats>(),plane);ck(cudaGetLastError());ck(cudaEventRecord(stop.e));
        // Completion is the ownership handoff: CPU must not read output or recycle input early.
        ck(cudaEventSynchronize(stop.e));float ms=0;ck(cudaEventElapsedTime(&ms,start.e,stop.e));kernel_ms+=ms;++launches;
        for(unsigned k=0;k<depth;++k){Stats s=initial();for(unsigned i=0;i<tiles;++i)append(s,partials.cpu<Stats>()[k*tiles+i]);need(!s.invalid && s.count==plane && s.first<plane,"nonfinite voxel or incomplete plane");rows[z+k]=s;}
    }
    a.finish();b.finish();x.check();y.check();partials.check();
    std::cout<<std::setprecision(17)<<"{\"format\":\"CFDK-GB10-mapped-residual-v1\",\"gpu\":\""<<p.name<<"\",\"shape_xyz\":["<<nx<<","<<ny<<","<<nz<<"],\"band\":"<<band<<",\"tile_size\":"<<Tile<<",\"partial_bytes\":"<<sizeof(Stats)<<",\"mapped_bytes\":"<<mapped<<",\"owned_total_bytes\":"<<all<<",\"budget_bytes\":"<<(size_t(budget)<<20)<<",\"read_bytes\":"<<bytes_read<<",\"h2d_bytes\":0,\"d2h_bytes\":0,\"launches\":"<<launches<<",\"guard_checks\":6,\"can_map_host\":"<<p.canMapHostMemory<<",\"uva\":"<<p.unifiedAddressing<<",\"pageable_access\":"<<pageable<<",\"uses_host_page_tables\":"<<directpt<<",\"host_register_supported\":"<<hostreg<<",\"read_ms\":"<<read_ms<<",\"kernel_ms\":"<<kernel_ms<<",\"wall_ms\":"<<milliseconds(begin)<<",\"rows\":[";
    for(unsigned z=0;z<nz;++z){const auto& s=rows[z];if(z)std::cout<<",";std::cout<<"["<<s.count<<","<<s.equal;for(auto v:s.over)std::cout<<","<<v;std::cout<<","<<s.maximum<<","<<s.first<<","<<s.bias<<","<<s.absolute<<","<<s.sse<<","<<s.reference<<"]";}std::cout<<"]}\n";
}
}
int main(int argc,char** argv){try{spark_mapped::run_mapped(argc,argv);return 0;}catch(const std::exception& e){std::cerr<<"mapped-residual: "<<e.what()<<"\n";return 2;}}
