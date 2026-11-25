// Query the real driver-loaded function, not PTX virtual register declarations.
#include <cuda.h>
#include <cstdio>
#include <stdexcept>
#include <string>
static void ck(CUresult r){ if(r!=CUDA_SUCCESS){const char *s=nullptr;cuGetErrorName(r,&s);throw std::runtime_error(s?s:"CUDA error");}}
int main(int argc,char**argv){try{
    if(argc!=2)throw std::runtime_error("PTX required");
    ck(cuInit(0));CUdevice d; ck(cuDeviceGet(&d,0));char name[128]={};ck(cuDeviceGetName(name,sizeof name,d));
    auto attr=[&](CUdevice_attribute x){int v;ck(cuDeviceGetAttribute(&v,x,d));return v;};
    if(std::string(name)!="NVIDIA GB10"||attr(CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR)!=12||attr(CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR)!=1)throw std::runtime_error("wrong GPU");
    int maxthr=attr(CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR);CUcontext ctx;ck(cuDevicePrimaryCtxRetain(&ctx,d));ck(cuCtxPushCurrent(ctx));CUmodule mod;ck(cuModuleLoad(&mod,argv[1]));
    for(const char *entry:{"back8","back","back32","back_roi8","back_roi","back_roi32"}){
        CUfunction f;ck(cuModuleGetFunction(&f,mod,entry));auto fn=[&](CUfunction_attribute x){int v;ck(cuFuncGetAttribute(&v,x,f));return v;};
        for(int threads:{64,128,256}){int blocks;ck(cuOccupancyMaxActiveBlocksPerMultiprocessor(&blocks,f,threads,0));
            std::printf("{\"entry\":\"%s\",\"threads\":%d,\"max_threads_per_sm\":%d,\"active_blocks_bound\":%d,\"registers\":%d,\"static_shared_bytes\":%d,\"local_bytes\":%d,\"binary_version\":%d}\n",entry,threads,maxthr,blocks,fn(CU_FUNC_ATTRIBUTE_NUM_REGS),fn(CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES),fn(CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES),fn(CU_FUNC_ATTRIBUTE_BINARY_VERSION));}}
    ck(cuModuleUnload(mod));CUcontext pop;ck(cuCtxPopCurrent(&pop));ck(cuDevicePrimaryCtxRelease(d));return 0;
}catch(const std::exception&e){std::fprintf(stderr,"%s\n",e.what());return 2;}}
