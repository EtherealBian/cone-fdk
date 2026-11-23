#pragma once
#include "../src/gpu.hpp" // Checked Windows ABI loader/types only. No cuFFT instance is created.
struct LabelsGpu {
    Lib d{"nvcuda.dll"};
#define DRV(n, ...)                                                                                \
    using n##_t = int (*)(__VA_ARGS__);                                                            \
    n##_t n = d.sym<n##_t>(#n)
    DRV(cuInit, unsigned);
    DRV(cuDeviceGet, int *, int);
    DRV(cuDevicePrimaryCtxRetain, Handle *, int);
    DRV(cuCtxSetCurrent, Handle);
    DRV(cuDevicePrimaryCtxRelease_v2, int);
    DRV(cuDeviceGetName, char *, int, int);
    DRV(cuModuleLoadData, Handle *, const void *);
    DRV(cuModuleUnload, Handle);
    DRV(cuModuleGetFunction, Handle *, Handle, const char *);
    DRV(cuMemGetInfo_v2, size_t *, size_t *);
    DRV(cuMemAlloc_v2, Ptr *, size_t);
    DRV(cuMemFree_v2, Ptr);
    DRV(cuMemcpyHtoD_v2, Ptr, const void *, size_t);
    DRV(cuMemcpyDtoH_v2, void *, Ptr, size_t);
    DRV(cuLaunchKernel, Handle, unsigned, unsigned, unsigned, unsigned, unsigned, unsigned,
        unsigned, Handle, void **, void **);
#undef DRV
    int dev = 0;
    Handle ctx{}, mod{}, classify{}, connect{}, resolve{};
    Ptr io{}, parents{};
    char name[128]{};
    LabelsGpu(const std::string &ptx) {
        CK(cuInit(0));
        CK(cuDeviceGet(&dev, 0));
        CK(cuDevicePrimaryCtxRetain(&ctx, dev));
        CK(cuCtxSetCurrent(ctx));
        CK(cuDeviceGetName(name, sizeof name, dev));
        CK(cuModuleLoadData(&mod, ptx.c_str()));
        CK(cuModuleGetFunction(&classify, mod, "classify"));
        CK(cuModuleGetFunction(&connect, mod, "connect"));
        CK(cuModuleGetFunction(&resolve, mod, "resolve"));
    }
    ~LabelsGpu() {
        if (parents)
            cuMemFree_v2(parents);
        if (io)
            cuMemFree_v2(io);
        if (mod)
            cuModuleUnload(mod);
        if (ctx)
            cuDevicePrimaryCtxRelease_v2(dev);
    }
    template <class... A> void run(Handle k, int n, A... a) {
        void *argv[] = {static_cast<void *>(&a)...};
        CK(cuLaunchKernel(k, (n + 255) / 256, 1, 1, 256, 1, 1, 0, nullptr, argv, nullptr));
    }
    void label(float *values, int *labels, int nx, int ny, int n, float threshold, int median,
               int conn) {
        CK(cuMemcpyHtoD_v2(io, values, size_t(n) * 4));
        run(classify, n, io, parents, nx, ny, n, threshold, median);
        run(connect, n, parents, nx, ny, n, conn);
        run(resolve, n, parents, io, n);
        CK(cuMemcpyDtoH_v2(labels, io, size_t(n) * 4));
    }
};
