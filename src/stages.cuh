#pragma once
#include "geometry.cuh"
#include "cuda_utils.cuh"
namespace fdk {
inline int ceildiv(int a, int b) {
    return (a - 1) / b + 1;
}
struct View {
    float c, s, ox, oy, angle_weight;
};
struct Stages {
    Geometry g;
    int k, rows;
    Device<float> detector, response;
    Device<float2> freq;
    Device<View> views;
    cufftHandle plan;
    Stages(Geometry, int batch, const std::vector<View> &);
    ~Stages() {
        cufftDestroy(plan);
    }
    void filter(int first, int actual);
    void backproject(float *volume, int first, int actual);
};
} // namespace fdk
