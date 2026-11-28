#pragma once
#include <cuda_runtime.h>
#include <cufft.h>
#include <stdexcept>
#include <string>
inline void checked(cudaError_t rc) {
    if (rc != cudaSuccess)
        throw std::runtime_error(cudaGetErrorString(rc));
}
inline void checked(cufftResult rc) {
    if (rc != CUFFT_SUCCESS)
        throw std::runtime_error("cuFFT status " + std::to_string(rc));
}
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
