#include "gpu.hpp"
#include <cuda.h>
#include <cstring>
// Keep CUDA header ABI macros out of the frozen FDK translation unit.
void spark_require_ats(Gpu& g){
    auto get=g.d.sym<int(*)(int*,int,int)>("cuDeviceGetAttribute");
    auto attr=[&](CUdevice_attribute a){int v=0;CK(get(&v,int(a),g.dev));return v;};
    if(std::strcmp(g.name,"NVIDIA GB10") || attr(CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR)!=12 || attr(CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR)!=1 || !attr(CU_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS) || !attr(CU_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS_USES_HOST_PAGE_TABLES) || !attr(CU_DEVICE_ATTRIBUTE_UNIFIED_ADDRESSING))
        throw std::runtime_error("qualified system-memory page-table access on GB10 required");
}
