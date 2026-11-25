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

// Shared point sampler for non-Cartesian rows and oriented sections.
__device__ __forceinline__ void accumulate_point(float *dst, const float2 *__restrict__ views,
                                                 const float2 *__restrict__ cs, Geom g, int first,
                                                 int cnt, float x, float y, float z) {
    float sum = *dst;
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
    *dst = sum;
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
    accumulate_point(out + j * plane.w + i, views, cs, g, first, cnt, xyz[0], xyz[1], xyz[2]);
}

extern "C" __global__ void back_cylinder(float *out, const float2 *__restrict__ views,
                                         const float2 *__restrict__ cs, Geom g, int first, int cnt,
                                         Cylinder cyl, int shell, int row0, int rows) {
    int i = blockIdx.x * blockDim.x + threadIdx.x, j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= cyl.w || j >= rows)
        return;
    double s, c;
    sincospi(2. * i / cyl.w, &s, &c);
    double t = (j + row0 - .5 * (cyl.h - 1)) * cyl.pitch;
    double radius = cyl.firstRadius + shell * cyl.radialStep;
    float xyz[3];
#pragma unroll
    for (int k = 0; k < 3; ++k)
        xyz[k] = (float)(cyl.center[k] + t * cyl.axis[k] +
                         radius * (c * cyl.seam[k] + s * cyl.tangent[k]));
    accumulate_point(out + j * cyl.w + i, views, cs, g, first, cnt, xyz[0], xyz[1], xyz[2]);
}

// Each lane follows one outward ray. All lanes in a warp read adjacent azimuths.
extern "C" __global__ void radial_wall(const float *tile, WallSample *out, int count, int shells,
                                       double r0, double dr, double level) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count)
        return;
    bool bad = false, ambiguous = false;
    double prior = tile[i], enter = 0, exit = 0;
    bool clipped = prior >= level;
    int runs = 0, foreground = 0;
    for (int k = 0; k < shells; ++k) {
        double v = tile[k * count + i];
        bad |= !isfinite(v);
        foreground += (v > level);
        if (k && k + 1 < shells && v == level) {
            double next = tile[(k + 1) * count + i];
            // An exact knot is unique only when its two neighbors straddle it.
            ambiguous |= !((prior < level && next > level) || (prior > level && next < level));
        }
        if (k && (v > level) != (prior > level)) {
            double edge = (k - 1) + (level - prior) / (v - prior);
            if (v > level) {
                enter = edge;
                ++runs;
            } else {
                exit = edge;
            }
        }
        prior = v;
    }
    clipped |= prior >= level;
    unsigned status = bad              ? Nonfinite
                      : clipped        ? Clipped
                      : ambiguous      ? Ambiguous
                      : runs > 1       ? Multiple
                      : runs == 0      ? Empty
                      : foreground < 2 ? Undersampled
                                       : Measured;
    double na = __longlong_as_double(0x7ff8000000000000LL);
    out[i] = status == Measured
                 ? WallSample{r0 + enter * dr, r0 + exit * dr, (exit - enter) * dr, status, 0}
                 : WallSample{na, na, na, status, 0};
}

// Deterministic block tree. Every lane reads the same normal equations before scratch reuse.
template <int N> __device__ void bore_sum(double (&v)[N], double scratch[][128]) {
    int t = threadIdx.x;
    for (int k = 0; k < N; ++k)
        scratch[k][t] = v[k];
    __syncthreads();
    for (int s = 64; s; s /= 2) {
        if (t < s)
            for (int k = 0; k < N; ++k)
                scratch[k][t] += scratch[k][t + s];
        __syncthreads();
    }
    for (int k = 0; k < N; ++k)
        v[k] = scratch[k][0];
    __syncthreads();
}

extern "C" __global__ void bore_circles(const WallSample *wall, BoreCircle *out, int w) {
    const int idx = blockIdx.x, t = threadIdx.x;
    const WallSample *row = wall + (idx / 2) * w;
    __shared__ double scratch[9][128];
    double init[4] = {};
    for (int i = t; i < w; i += 128) {
        if (row[i].status != Measured)
            continue;
        double r = (idx & 1) ? row[i].outer : row[i].inner, s, c;
        sincospi(2. * i / w, &s, &c);
        init[0]++;
        init[1] += r;
        init[2] += r * c;
        init[3] += r * s;
    }
    bore_sum(init, scratch);
    BoreCircle fit = {};
    fit.valid = int(init[0]);
    // No gap filling or extrapolation: a single unmeasured ray suppresses the row.
    if (fit.valid == w) {
        const double scale = init[1] / w;
        // The first harmonic is the exact center for an even, uniform sampling of a circle.
        double cx = (init[2] / init[1]) * 2, cy = (init[3] / init[1]) * 2;
        for (int it = 0; it < 16; ++it) {
            double m[9] = {};
            for (int i = t; i < w; i += 128) {
                double r = ((idx & 1) ? row[i].outer : row[i].inner) / scale, s, c;
                sincospi(2. * i / w, &s, &c);
                double x = cx - r * c, y = cy - r * s, d = hypot(x, y);
                if (!(d > 1e-15)) {
                    m[8]++;
                    continue;
                }
                double u = x / d, v = y / d;
                m[0] += u;
                m[1] += v;
                m[2] += d;
                m[3] += u * u;
                m[4] += u * v;
                m[5] += v * v;
                m[6] += x;
                m[7] += y;
            }
            bore_sum(m, scratch);
            // Eliminate the freely fitted radius (mean distance) from the 3x3 system.
            double radius = m[2] / w, a = m[3] - m[0] * m[0] / w, b = m[4] - m[0] * m[1] / w,
                   c = m[5] - m[1] * m[1] / w, det = a * c - b * b, q = radius * m[0] - m[6],
                   p = radius * m[1] - m[7];
            if (m[8] || !(det > 1e-12 * w * w))
                break;
            double dx = (c * q - b * p) / det, dy = (a * p - b * q) / det;
            if (hypot(dx, dy) <= 1e-13) {
                double ss[1] = {}, peak = 0;
                for (int i = t; i < w; i += 128) {
                    double r = ((idx & 1) ? row[i].outer : row[i].inner) / scale, sn, cs;
                    sincospi(2. * i / w, &sn, &cs);
                    double e = hypot(cx - r * cs, cy - r * sn) - radius;
                    peak = fmax(peak, fabs(e));
                    ss[0] += e * e;
                }
                bore_sum(ss, scratch);
                scratch[0][t] = peak;
                __syncthreads();
                for (int n = 64; n; n /= 2) {
                    if (t < n)
                        scratch[0][t] = fmax(scratch[0][t], scratch[0][t + n]);
                    __syncthreads();
                }
                fit.x = cx * scale;
                fit.y = cy * scale;
                fit.radius = radius * scale;
                fit.rms = sqrt(ss[0] / w) * scale;
                fit.peak = scratch[0][0] * scale;
                fit.solved = 1;
                break;
            }
            cx += dx;
            cy += dy;
        }
    }
    if (!t)
        out[idx] = fit;
}

// A nonnegative label is a live union-find node. Rejected pairs retain BOTH statuses.
extern "C" __global__ void loss_classify(const WallSample *ref, const WallSample *part, int *parent,
                                         int n, double threshold) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;
    unsigned a = ref[i].status, b = part[i].status;
    parent[i] = (a || b) ? -2 - int(a * WallStatusCount + b)
                         : (ref[i].thickness - part[i].thickness >= threshold ? i : -1);
}
// Every successful link strictly lowers a root. A lost CAS retries from the new roots;
// a tile thus has a deterministic minimum-index representative, regardless of launch order.
__device__ int loss_root(int *p, int i) {
    while (i >= 0) {
        int j = atomicAdd(p + i, 0);
        if (j < 0 || j == i)
            return j;
        int grand = atomicAdd(p + j, 0);
        if (grand != j)
            atomicCAS(p + i, j, grand); // path halving, never overwrite a newer link
        i = grand;
    }
    return i;
}
__device__ void loss_join(int *p, int a, int b) {
    for (;;) {
        a = loss_root(p, a);
        b = loss_root(p, b);
        if (a < 0 || b < 0 || a == b)
            return;
        int hi = max(a, b), lo = min(a, b);
        if (atomicCAS(p + hi, hi, lo) == hi)
            return;
    }
}
extern "C" __global__ void loss_connect(int *parent, int w, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n || atomicAdd(parent + i, 0) < 0)
        return;
    loss_join(parent, i, i % w ? i - 1 : i + w - 1);
    if (i >= w)
        loss_join(parent, i, i - w);
}
