#pragma once
#include <cuda_runtime.h>
#include <cufft.h>
#include <stdexcept>
#include <string>
inline void checked(cudaError_t rc) {
    if (rc != cudaSuccess)
        throw std::runtime_error(cudaGetErrorString(rc));
}
inline void fft_checked(cufftResult rc) {
    if (rc != CUFFT_SUCCESS)
        throw std::runtime_error("cuFFT status " + std::to_string(rc));
}
#ifndef NDEBUG
#define CUDA_CHECK(call) checked(call)
#define CUDA_CHECK_LAST() checked(cudaGetLastError())
#else
#define CUDA_CHECK(call) static_cast<void>(call)
#define CUDA_CHECK_LAST() ((void)0)
#endif
// Unique owners: copying these handles would double-free device resources.
template <class T> struct Device {
    T *p = nullptr;
    explicit Device(size_t n) {
        checked(cudaMalloc(&p, n * sizeof(T)));
    }
    ~Device() {
        cudaFree(p);
    }
    Device(const Device &) = delete;
    Device &operator=(const Device &) = delete;
};
struct Events {
    struct Event {
        cudaEvent_t v;
        Event() {
            checked(cudaEventCreate(&v));
        }
        ~Event() {
            cudaEventDestroy(v);
        }
        Event(const Event &) = delete;
        Event &operator=(const Event &) = delete;
    } start, end;
    void begin() {
        checked(cudaEventRecord(start.v));
    }
    float finish() {
        float ms;
        checked(cudaEventRecord(end.v));
        checked(cudaEventSynchronize(end.v));
        checked(cudaEventElapsedTime(&ms, start.v, end.v));
        return ms;
    }
};
