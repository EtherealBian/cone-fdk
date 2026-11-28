#include "stages.cuh"

/**
 * back — Accumulate one angle batch into a dense cubic volume.
 *
 * Thread mapping: 16 x 8 threads per block; one (x,y) and up to 16 z values per thread.
 * Memory: Register accumulators, global angle and detector tables, coalesced volume rows.
 * Occupancy: 58.02% achieved; 72 registers/thread (GB10, first launch).
 * Known bottleneck: See optimization-log, Z tiling and detector access.
 */
extern "C" __global__ void back(float *__restrict__ out, const float2 *__restrict__ views,
                                const float2 *__restrict__ cs, Geometry g, int first, int cnt) {
    constexpr int lanes = 16;
    int ix = blockIdx.x * blockDim.x + threadIdx.x, iy = blockIdx.y * blockDim.y + threadIdx.y,
        iz = int(blockIdx.z) * lanes;
    if (ix >= g.n || iy >= g.n || iz >= g.n)
        return;
    float mid = (g.n - 1) * .5f, x = (ix - mid) * g.vox, y = (iy - mid) * g.vox;
    unsigned long long plane = (unsigned long long)g.n * g.n, pixel = iy * g.n + ix;
    float val[lanes];
#pragma unroll
    for (int j = 0; j < lanes; j++)
        val[j] = iz + j < g.n ? out[pixel + (iz + j) * plane] : 0;
    for (int a = 0; a < cnt; a++) {
        float2 b = cs[first + a];
        float r = 1.f / (g.R - x * b.y + y * b.x), m = g.D * r;
        float u = fmaf((x * b.x + y * b.y), m / g.du, (g.u - 1) * .5f);
        int ui = __float2int_rd(u);
        float fu = u - ui;
        if (ui < -1 || ui >= g.u)
            continue;
        int stride = g.u + 1, pixels = g.v * stride;
        const float2 *p = views + (unsigned long long)a * (pixels + 2) + 1;
        float w = g.R * r;
        w *= w;
#pragma unroll
        for (int j = 0; j < lanes; j++) {
            // Keep each original four-plane group's coordinate evaluation order;
            // regrouping (base + j*vox) changes rounding at some pitches.
            float z = (iz + (j & ~3) - mid) * g.vox;
            float v = fmaf(z + (j & 3) * g.vox, m / g.dv, (g.v - 1) * .5f);
            int vi = __float2int_rd(v);
            float fv = v - vi;
            // Clamp before multiplication as valid geometry can still project far off-panel.
            int i = max(-2, min(g.v, vi)) * stride + (ui + 1);
            float2 lo = p[max(-1, min(pixels, i))], hi = p[max(-1, min(pixels, i + stride))];
            float s = fmaf(fu, lo.y, lo.x), t = fmaf(fu, hi.y, hi.x);
            val[j] = fmaf(fmaf(fv, t - s, s), w, val[j]);
        }
    }
#pragma unroll
    for (int j = 0; j < lanes; j++)
        if (iz + j < g.n)
            out[pixel + (iz + j) * plane] = val[j];
}

void fdk::Stages::backproject(float *out, int first, int actual) {
    dim3 grid(ceildiv(g.n, 16), ceildiv(g.n, 8), ceildiv(g.n, 16));
    back<<<grid, dim3(16, 8)>>>(out, freq.p, cs.p, g, first, actual);
    CUDA_CHECK_LAST();
}
