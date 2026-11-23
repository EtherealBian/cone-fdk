// Circular, equiangular flat-panel cone-beam FDK. Input views are [v,u].
// Native CUDA C++ (compiled to PTX at build time; no Python on the hot path).
struct Geom {
    int a, v, u, n;
    float du, dv, R, D, vox, start, sign;
};
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
extern "C" __global__ void back(float *out, const float2 *__restrict__ views,
                                const float2 *__restrict__ cs, Geom g, int first, int cnt, int z0,
                                int depth) {
    constexpr int lanes = 16;
    int ix = blockIdx.x * blockDim.x + threadIdx.x, iy = blockIdx.y * blockDim.y + threadIdx.y,
        iz = blockIdx.z * lanes;
    if (ix >= g.n || iy >= g.n || iz >= depth)
        return;
    float mid = (g.n - 1) * .5f, x = (ix - mid) * g.vox, y = (iy - mid) * g.vox;
    unsigned long long plane = (unsigned long long)g.n * g.n, off = plane * iz + iy * g.n + ix;
    float val[lanes];
#pragma unroll
    for (int j = 0; j < lanes; j++)
        val[j] = iz + j < depth ? out[off + j * plane] : 0;
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
        if (iz + j < depth)
            out[off + j * plane] = val[j];
}
