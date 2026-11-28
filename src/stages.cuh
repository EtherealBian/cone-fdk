#pragma once
#include "geometry.cuh"
#include "cuda_utils.cuh"
namespace fdk {
inline int ceildiv(int a, int b) {
    return (a - 1) / b + 1;
}
struct Stages {
    Geometry g;
    int k, rows, entries;
    Device<float> detector, response;
    Device<float2> freq, cs;
    cufftHandle plan;
    Stages(Geometry geom, int batch_size);
    ~Stages() {
        cufftDestroy(plan);
    }
    void filter(int actual);
    void backproject(float *out, int first, int actual);
};
} // namespace fdk
