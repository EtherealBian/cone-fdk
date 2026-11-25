// Circular, equiangular flat-panel cone-beam FDK. Input views are [v,u].
// Native CUDA C++ (compiled to PTX at build time; no Python on the hot path).
#include "geometry.hpp"
extern "C" __global__ void angles(float2 *cs, Geom g) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < g.a) {
        float c, s;
        sincosf(g.start + g.sign * 6.283185307179586f * i / g.a, &s, &c);
        cs[i] = make_float2(c, s);
    }
}
// DFT of the finite, symmetric spatial Ram-Lak impulse. Nonzero discrete DC
// matters on small detectors; |f| sampled directly is not the same filter.
extern "C" __global__ void ramp(float *dst, int k) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= k)
        return;
    double h = .25;
    for (int j = 1; j < k / 2; j += 2)
        h -= 2. * cos(6.2831853071795864769 * i * j / k) / (9.869604401089358619 * j * j);
    dst[i] = (float)(h / k);
}
extern "C" __global__ void pre(const float *in, float2 *f, Geom g, int k, int actual, int rows) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= rows * k)
        return;
    int x = t % k, row = t / k;
    float p = 0;
    if (x < g.u && row < actual * g.v) {
        float u = (x - (g.u - 1) * .5f) * g.du, v = (row % g.v - (g.v - 1) * .5f) * g.dv;
        p = in[row * g.u + x] * (g.D * rsqrtf(g.D * g.D + u * u + v * v));
    }
    f[t] = make_float2(p, 0);
}
extern "C" __global__ void spectrum(float2 *f, const float *h, int k, int total) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t < total) {
        float w = h[t % k];
        f[t].x *= w;
        f[t].y *= w;
    }
}
extern "C" __global__ void unpack(const float2 *f, float *out, Geom g, int k, int total) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t < total)
        out[t] = f[(t / g.u) * k + t % g.u].x * (3.141592653589793f / g.a) * (g.D / (g.R * g.du));
}
// Pack horizontal interpolation into (left, right-left) once per detector sample,
// not once per voxel. Reuse the now-dead complex FFT allocation. Each view
// has a virtual left column and two zero guard entries for vertical clipping.
extern "C" __global__ void pairs(const float *__restrict__ in, float2 *__restrict__ lut, int width,
                                 int height, int entries, int total) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= total)
        return;
    int a = t / entries, pixel = t % entries - 1;
    float left = 0.f, right = 0.f;
    if ((unsigned)pixel < (unsigned)(entries - 2)) {
        int y = pixel / (width + 1), u = pixel % (width + 1) - 1;
        const float *row = in + (a * height + y) * width;
        if (u >= 0)
            left = row[u];
        if (u + 1 < width)
            right = row[u + 1];
    }
    lut[t] = make_float2(left, right - left);
}
// Adjacent z planes reuse the same cone denominator and detector column.
// Border interpolation is zero extended, not clamped to the nearest pixel.
template <bool roi>
__device__ __forceinline__ void back_tile(float *out, const float2 *__restrict__ views,
                                          const float2 *__restrict__ cs, Geom g, int first, int cnt,
                                          int z0, int depth, int x0, int y0, int nx, int ny) {
    constexpr int lanes = 16;
    int ix = blockIdx.x * blockDim.x + threadIdx.x, iy = blockIdx.y * blockDim.y + threadIdx.y,
        // Start on the reference grid's four-plane rounding group. Up to three
        // leading lanes of the first ROI tile are inactive (also at slab seams).
        iz = int(blockIdx.z) * lanes - (roi ? (z0 & 3) : 0);
    if (ix >= nx || iy >= ny || iz >= depth)
        return;
    float mid = (g.n - 1) * .5f, x = (ix + x0 - mid) * g.vox, y = (iy + y0 - mid) * g.vox;
    unsigned long long plane = (unsigned long long)nx * ny, pixel = iy * nx + ix;
    float val[lanes];
#pragma unroll
    for (int j = 0; j < lanes; j++)
        val[j] = (!roi || iz + j >= 0) && iz + j < depth ? out[pixel + (iz + j) * plane] : 0;
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
            float z = (z0 + iz + (j & ~3) - mid) * g.vox;
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
        if ((!roi || iz + j >= 0) && iz + j < depth)
            out[pixel + (iz + j) * plane] = val[j];
}

extern "C" __global__ void back(float *out, const float2 *__restrict__ views,
                                const float2 *__restrict__ cs, Geom g, int first, int cnt, int z0,
                                int depth) {
    back_tile<false>(out, views, cs, g, first, cnt, z0, depth, 0, 0, g.n, g.n);
}

extern "C" __global__ void back_roi(float *out, const float2 *__restrict__ views,
                                    const float2 *__restrict__ cs, Geom g, int first, int cnt,
                                    int z0, int depth, int x0, int y0, int nx, int ny) {
    back_tile<true>(out, views, cs, g, first, cnt, z0, depth, x0, y0, nx, ny);
}

// An arbitrary plane cannot reuse a Z column. One thread accumulates one pixel;
// only the world-space pixel address is computed in FP64, once per view batch.
extern "C" __global__ void back_plane(float *out, const float2 *__restrict__ views,
                                      const float2 *__restrict__ cs, Geom g, int first, int cnt,
                                      Plane plane, int row0, int rows) {
    int i = blockIdx.x * blockDim.x + threadIdx.x, j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= plane.w || j >= rows)
        return;
    double di = i - .5 * (plane.w - 1), dj = j + row0 - .5 * (plane.h - 1);
    float xyz[3];
#pragma unroll
    for (int k = 0; k < 3; ++k)
        xyz[k] = (float)(plane.center[k] + di * plane.col[k] + dj * plane.row[k]);
    float x = xyz[0], y = xyz[1], z = xyz[2], sum = out[j * plane.w + i];
    int stride = g.u + 1, pixels = g.v * stride;
    for (int a = 0; a < cnt; ++a) {
        float2 b = cs[first + a];
        float r = 1.f / (g.R - x * b.y + y * b.x), m = g.D * r;
        float u = fmaf(x * b.x + y * b.y, m / g.du, (g.u - 1) * .5f);
        float v = fmaf(z, m / g.dv, (g.v - 1) * .5f);
        if (!(u > -1.f && u < g.u && v > -1.f && v < g.v))
            continue;
        int ui = __float2int_rd(u), vi = __float2int_rd(v), index = vi * stride + ui + 1;
        const float2 *p = views + (unsigned long long)a * (pixels + 2) + 1;
        float2 lo = p[max(-1, index)], hi = p[min(pixels, index + stride)];
        float fu = u - ui, fv = v - vi, w = g.R * r;
        float s = fmaf(fu, lo.y, lo.x), t = fmaf(fu, hi.y, hi.x);
        sum = fmaf(fmaf(fv, t - s, s), w * w, sum);
    }
    out[j * plane.w + i] = sum;
}
