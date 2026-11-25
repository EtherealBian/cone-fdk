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
// Physical frame for a cylindrical sheet. Host and CUDA share this layout.
struct Cylinder {
    int w, h;
    double center[3], axis[3], seam[3], tangent[3];
    double firstRadius, radialStep, pitch;
};
static_assert(sizeof(Cylinder) == 128, "cylinder kernel ABI");

// Little-endian CFDK wall map v1: also the on-disk record. No implicit padding.
enum WallStatus : unsigned {
    Measured,
    Empty,
    Clipped,
    Multiple,
    Ambiguous,
    Nonfinite,
    Undersampled,
    WallStatusCount
};
struct WallSample {
    double inner, outer, thickness;
    unsigned status, reserved;
};
static_assert(sizeof(WallSample) == 32, "wall map ABI");
