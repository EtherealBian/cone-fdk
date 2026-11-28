#pragma once
#include <vector>
struct Geometry {
    int a, v, u, n;
    float du, dv, R, D, vox, start, sign;
};
using Projections = std::vector<float>;
using Volume = std::vector<float>;
void fdk_reconstruct(const Projections &proj, const Geometry &geo, Volume &out);
