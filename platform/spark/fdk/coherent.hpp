#pragma once
// Intended for exactly the checked Spark dense/single-ROI product; same kernel, same address offsets.
#include <cstring>
#include <sys/mman.h>
extern void spark_require_ats(Gpu&);
struct SharedBuf {
    Gpu& g; const size_t sz; const bool shared; Ptr base=0,p=0;
    std::unique_ptr<Buf> staged;
    SharedBuf(Gpu& gpu,size_t n,bool use):g(gpu),sz(n),shared(use) {
        if(!shared){staged=std::make_unique<Buf>(g,n);p=staged->p;return;}
        void* host=mmap(nullptr,n+2*Buf::pad,PROT_READ|PROT_WRITE,MAP_PRIVATE|MAP_ANONYMOUS,-1,0);
        if(host==MAP_FAILED) throw std::runtime_error("system ATS allocation failed");
        base=reinterpret_cast<uintptr_t>(host);p=base+Buf::pad;
        auto first=static_cast<unsigned char*>(host);
        for(size_t i=0;i<Buf::pad;i+=4){unsigned pattern=Buf::pattern;std::memcpy(first+i,&pattern,4);std::memcpy(first+Buf::pad+n+i,&pattern,4);}
        g.live+=n+2*Buf::pad;g.peak=std::max(g.peak,g.live);
    }
    float* cpu() const {
        if(!shared) throw std::runtime_error("CPU access requested to device storage");
        return reinterpret_cast<float*>(static_cast<uintptr_t>(p));
    }
    void verify(){
        if(!shared){staged->verify();return;}
        // Caller synchronizes first. Both redzones are inspected directly by the CPU.
        for(size_t i=0;i<Buf::pad/sizeof(unsigned);++i){
            unsigned a,b; std::memcpy(&a,reinterpret_cast<const char*>(cpu())-Buf::pad+i*4,4);
            std::memcpy(&b,reinterpret_cast<const char*>(cpu())+sz+i*4,4);
            if(a!=Buf::pattern || b!=Buf::pattern) throw std::runtime_error("shared red zone was overwritten");
        }
    }
    ~SharedBuf(){if(base){munmap(reinterpret_cast<void*>(static_cast<uintptr_t>(base)),sz+2*Buf::pad);g.live-=sz+2*Buf::pad;}}
    SharedBuf(const SharedBuf&)=delete;
};
