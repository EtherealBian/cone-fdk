#pragma once
// Passed by value via the CUDA driver launch ABI; NVRTC receives this same header.
struct Geom {
    int a, v, u, n;
    float du, dv, R, D, vox, start, sign;
};
struct Plane {
    int w, h;
    double center[3], col[3], row[3];
};
static_assert(sizeof(Geom) == 44 && sizeof(Plane) == 80, "kernel ABI");
