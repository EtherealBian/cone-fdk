#pragma once
#include <vector>

struct Geometry {
    int a, v, u;
    float du, dv, R, D;
    static constexpr int n = 256;
    static constexpr float vox = 1.25f; // mm; centered on the RTK isocenter
};
struct Pose {
    double degrees, offset_x, offset_y;
};
using Projections = std::vector<float>;
using Volume = std::vector<float>;
// Corrected [view,row,column] line integrals in; RTK [z,y,x] volume out.
// Returns summed CUDA event milliseconds for just the backprojection kernels.
float fdk_reconstruct(const Projections &, const Geometry &, const std::vector<Pose> &, Volume &);
