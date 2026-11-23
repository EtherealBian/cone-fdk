#pragma once
// Windows x64 CUDA driver + cuFFT ABI subset. No SDK import library needed.
// All calls are resolved by name and status checked; runtime ABI is CUDA 12 cuFFT.
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <algorithm>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
#include <windows.h>
using Ptr = unsigned long long;
using Handle = void *;
inline void check(int rc, const char *call) {
    if (rc)
        throw std::runtime_error(std::string(call) + " failed (status " + std::to_string(rc) + ")");
}
#define CK(x) check((x), #x)
struct Lib {
    HMODULE h;
    explicit Lib(const char *name) : h(LoadLibraryA(name)) {
        if (!h)
            throw std::runtime_error(std::string("cannot load ") + name + " (Windows error " +
                                     std::to_string(GetLastError()) + ")");
    }
    template <class T> T sym(const char *s) {
        auto p = GetProcAddress(h, s);
        if (!p)
            throw std::runtime_error(std::string("missing symbol: ") + s);
        return reinterpret_cast<T>(p);
    }
};
struct Gpu {
    Lib d{"nvcuda.dll"}, f{"cufft64_11.dll"};
#define FN(lib, name, ...)                                                                         \
    using name##_t = int (*)(__VA_ARGS__);                                                         \
    name##_t name = lib.sym<name##_t>(#name)
    FN(d, cuInit, unsigned);
    FN(d, cuDeviceGet, int *, int);
    FN(d, cuDevicePrimaryCtxRetain, Handle *, int);
    FN(d, cuCtxSetCurrent, Handle);
    FN(d, cuDeviceGetName, char *, int, int);
    FN(d, cuDriverGetVersion, int *);
    FN(d, cuCtxSynchronize);
    FN(d, cuMemGetInfo_v2, size_t *, size_t *);
    FN(d, cuMemAlloc_v2, Ptr *, size_t);
    FN(d, cuMemFree_v2, Ptr);
    FN(d, cuMemAllocHost_v2, void **, size_t);
    FN(d, cuMemFreeHost, void *);
    FN(d, cuMemcpyHtoD_v2, Ptr, const void *, size_t);
    FN(d, cuMemcpyDtoH_v2, void *, Ptr, size_t);
    FN(d, cuMemsetD32_v2, Ptr, unsigned, size_t);
    FN(d, cuModuleLoadData, Handle *, const void *);
    FN(d, cuModuleUnload, Handle);
    FN(d, cuDevicePrimaryCtxRelease_v2, int);
    FN(d, cuModuleGetFunction, Handle *, Handle, const char *);
    FN(d, cuLaunchKernel, Handle, unsigned, unsigned, unsigned, unsigned, unsigned, unsigned,
       unsigned, Handle, void **, void **);
    FN(d, cuEventCreate, Handle *, unsigned);
    FN(d, cuEventRecord, Handle, Handle);
    FN(d, cuEventSynchronize, Handle);
    FN(d, cuEventElapsedTime, float *, Handle, Handle);
    FN(d, cuEventDestroy_v2, Handle);
    FN(f, cufftCreate, int *);
    FN(f, cufftSetAutoAllocation, int, int);
    FN(f, cufftMakePlan1d, int, int, int, int, size_t *);
    FN(f, cufftSetWorkArea, int, void *);
    FN(f, cufftExecC2C, int, void *, void *, int);
    FN(f, cufftDestroy, int);
    FN(f, cufftGetVersion, int *);
#undef FN
    Handle mod = nullptr, ctx = nullptr;
    char name[128] = {};
    int driver = 0, fftver = 0, dev = 0;
    size_t live = 0, peak = 0;
    explicit Gpu(const char *ptx) {
        CK(cuInit(0));

        CK(cuDeviceGet(&dev, 0));
        CK(cuDevicePrimaryCtxRetain(&ctx, dev));
        CK(cuCtxSetCurrent(ctx));
        CK(cuDeviceGetName(name, 128, dev));
        CK(cuDriverGetVersion(&driver));
        CK(cufftGetVersion(&fftver));
        CK(cuModuleLoadData(&mod, ptx));
    }
    ~Gpu() {
        if (mod)
            cuModuleUnload(mod);
        if (ctx)
            cuDevicePrimaryCtxRelease_v2(dev);
    }
    Handle kernel(const char *s) {
        Handle h;
        CK(cuModuleGetFunction(&h, mod, s));
        return h;
    }
    template <class... T> void launch(Handle k, int gx, int gy, int gz, int bx, int by, T... args) {
        void *argv[] = {static_cast<void *>(&args)...};
        CK(cuLaunchKernel(k, gx, gy, gz, bx, by, 1, 0, nullptr, argv, nullptr));
    }
};
// Bounded page-locked staging, not part of the device-buffer budget. Only
// CUDA_ERROR_OUT_OF_MEMORY is recoverable; never hide a prior GPU failure.
struct HostBuffer {
    Gpu &gpu;
    const size_t count;
    void *locked = nullptr;
    std::vector<float> fallback;
    HostBuffer(Gpu &g, size_t n, bool allowPin) : gpu(g), count(n) {
        if (allowPin && n * sizeof(float) <= (8u << 20)) {
            void *ptr = nullptr;
            int rc = gpu.cuMemAllocHost_v2(&ptr, n * sizeof(float));
            if (!rc)
                locked = ptr;
            else if (rc != 2 /* CUDA_ERROR_OUT_OF_MEMORY */)
                check(rc, "cuMemAllocHost_v2");
        }
        if (!locked)
            fallback.resize(n);
    }
    float *data() {
        return locked ? static_cast<float *>(locked) : fallback.data();
    }
    size_t pinnedBytes() const {
        return locked ? count * sizeof(float) : 0;
    }
    ~HostBuffer() {
        if (locked)
            gpu.cuMemFreeHost(locked);
    }
    HostBuffer(const HostBuffer &) = delete;
};
// Two aligned sentinels surround every nonempty allocation, including FFT scratch.
// They detect writes into the borders, not arbitrary OOB reads; this is not a sanitizer.
struct Buf {
    static constexpr size_t pad = 256;
    static constexpr unsigned pattern = 0x51A7C0DEu;
    Gpu &g;
    Ptr base = 0, p = 0;
    size_t sz;
    Buf(Gpu &g_, size_t n) : g(g_), sz(n) {
        if (n) {
            CK(g.cuMemAlloc_v2(&base, n + 2 * pad));
            p = base + pad;
            g.live += n + 2 * pad;
            g.peak = std::max(g.live, g.peak);
            CK(g.cuMemsetD32_v2(base, pattern, pad / 4));
            CK(g.cuMemsetD32_v2(p + n, pattern, pad / 4));
        }
    }
    void verify() {
        if (!base)
            return;
        unsigned before[pad / 4], after[pad / 4];
        CK(g.cuMemcpyDtoH_v2(before, base, pad));
        CK(g.cuMemcpyDtoH_v2(after, p + sz, pad));
        for (size_t i = 0; i < pad / 4; i++)
            if (before[i] != pattern || after[i] != pattern)
                throw std::runtime_error("GPU allocation red zone was overwritten");
    }
    ~Buf() {
        if (base) {
            g.cuMemFree_v2(base);
            g.live -= sz + 2 * pad;
        }
    }
    Buf(const Buf &) = delete;
};
struct Plan {
    Gpu &g;
    int id = 0;
    size_t work = 0;
    Plan(Gpu &g_, int k, int rows) : g(g_) {
        CK(g.cufftCreate(&id));
        try {
            CK(g.cufftSetAutoAllocation(id, 0));
            CK(g.cufftMakePlan1d(id, k, 0x29, rows, &work));
        } catch (...) {
            g.cufftDestroy(id);
            throw;
        }
    }
    ~Plan() {
        g.cufftDestroy(id);
    }
    Plan(const Plan &) = delete;
};
struct Timer {
    Gpu &g;
    Handle a, b;
    Timer(Gpu &v) : g(v) {
        CK(g.cuEventCreate(&a, 0));
        CK(g.cuEventCreate(&b, 0));
    }
    void start() {
        CK(g.cuEventRecord(a, nullptr));
    }
    double stop() {
        float ms;
        CK(g.cuEventRecord(b, nullptr));
        CK(g.cuEventSynchronize(b));
        CK(g.cuEventElapsedTime(&ms, a, b));
        return ms;
    }
    ~Timer() {
        g.cuEventDestroy_v2(a);
        g.cuEventDestroy_v2(b);
    }
};
