#include "stages.cuh"

extern "C" __global__ void back(float *__restrict__ out, const float2 *__restrict__ det,
                                const fdk::View *__restrict__ views, Geometry g, int count) {
    // RTK fixed axes: rotation about Y; X,Z form the gantry plane. The output
    // is conventional C-order [Z,Y,X], with neighboring threads writing adjacent Xs.
    constexpr int lanes = 16;
    int ix = blockIdx.x * blockDim.x + threadIdx.x, iz = blockIdx.y * blockDim.y + threadIdx.y;
    int y0 = int(blockIdx.z) * lanes;
    if (ix >= g.n || iz >= g.n)
        return;
    constexpr float mid = (Geometry::n - 1) * .5f;
    float x = (ix - mid) * g.vox, z = (iz - mid) * g.vox;
    float *column = out + (iz * g.n + y0) * g.n + ix;
    float sum[lanes];
#pragma unroll
    for (int j = 0; j < lanes; ++j)
        sum[j] = column[j * g.n];
    for (int i = 0; i < count; ++i) {
        const auto p = views[i];
        float inv = 1.f / (g.R - (p.s * x + p.c * z));
        float mag = g.D * inv;
        float u = ((p.c * x - p.s * z) * mag - p.ox) / g.du + (g.u - 1) * .5f;
        int ui = __float2int_rd(u);
        if (ui < 0 || ui >= g.u - 1)
            continue;
        float fu = u - ui, w = g.R * inv;
        const float2 *row0 = det + i * g.u * g.v + ui;
#pragma unroll
        for (int j = 0; j < lanes; ++j) {
            float v = (((y0 + j - mid) * g.vox) * mag - p.oy) / g.dv + (g.v - 1) * .5f;
            int vi = __float2int_rd(v);
            if (vi < 0 || vi >= g.v - 1)
                continue;
            float2 lo = row0[vi * g.u], hi = row0[(vi + 1) * g.u];
            float a = fmaf(fu, lo.y, lo.x), b = fmaf(fu, hi.y, hi.x);
            sum[j] = fmaf(fmaf(v - vi, b - a, a), w * w, sum[j]);
        }
    }
#pragma unroll
    for (int j = 0; j < lanes; ++j)
        column[j * g.n] = sum[j];
}

void fdk::Stages::backproject(float *out, int first, int actual) {
    static_assert(Geometry::n % 16 == 0, "Y tiles must cover the fixed grid exactly");
    dim3 grid(ceildiv(g.n, 16), ceildiv(g.n, 8), g.n / 16);
    back<<<grid, dim3(16, 8)>>>(out, freq.p, views.p + first, g, actual);
    checked(cudaGetLastError());
}
