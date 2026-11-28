#include "stages.cuh"
using fdk::View;
// DFT of the finite, symmetric spatial Ram-Lak impulse. Nonzero discrete DC
// matters on small detectors; |f| sampled directly is not the same filter.
extern "C" __global__ void ramp(float *__restrict__ dst, int k) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= k)
        return;
    double h = .25;
    for (int j = 1; j < k / 2; j += 2)
        h -= 2. * cos(6.2831853071795864769 * i * j / k) / (9.869604401089358619 * j * j);
    dst[i] = (float)(h / k);
}
extern "C" __global__ void pre(const float *__restrict__ in, float2 *__restrict__ f, Geometry g,
                               int k, int actual, int rows, const View *views) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= rows * k)
        return;
    int x = t % k, row = t / k;
    float p = 0;
    if (x < g.u && row < actual * g.v) {
        const View view = views[row / g.v];
        float u = (x - (g.u - 1) * .5f) * g.du + view.ox;
        float v = (row % g.v - (g.v - 1) * .5f) * g.dv + view.oy;
        p = in[row * g.u + x] * (g.D * rsqrtf(g.D * g.D + u * u + v * v));
    }
    f[t] = make_float2(p, 0);
}
extern "C" __global__ void spectrum(float2 *__restrict__ f, const float *__restrict__ h, int k,
                                    int total) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= total)
        return;
    float w = h[t % k];
    f[t].x *= w;
    f[t].y *= w;
}
extern "C" __global__ void unpack(const float2 *__restrict__ f, float *__restrict__ out, Geometry g,
                                  int k, int total, const View *views) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= total)
        return;
    out[t] =
        f[(t / g.u) * k + t % g.u].x * views[t / (g.u * g.v)].angle_weight * (g.D / (g.R * g.du));
}
// Precompute the two horizontal interpolation coefficients in the spent FFT workspace.
extern "C" __global__ void pairs(const float *in, float2 *lut, int width, int total) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t < total) {
        float left = in[t];
        lut[t] = make_float2(left, t % width + 1 < width ? in[t + 1] - left : 0.f);
    }
}

namespace fdk {
static int fft_width(int u) {
    int k = 1;
    while (k < 2 * u)
        k *= 2;
    return k;
}
Stages::Stages(Geometry geom, int b, const std::vector<View> &host)
    : g(geom), k(fft_width(g.u)), rows(b * g.v), detector(size_t(rows) * g.u), response(k),
      freq(size_t(rows) * k), views(g.a) {
    checked(cudaMemcpy(views.p, host.data(), host.size() * sizeof(View), cudaMemcpyHostToDevice));
    ramp<<<ceildiv(k, 128), 128>>>(response.p, k);
    checked(cudaGetLastError());
    checked(cufftPlan1d(&plan, k, CUFFT_C2C, rows));
}
void Stages::filter(int first, int actual) {
    const int nt = rows * k, nout = actual * g.v * g.u;
    pre<<<ceildiv(nt, 256), 256>>>(detector.p, freq.p, g, k, actual, rows, views.p + first);
    checked(cudaGetLastError());
    checked(cufftExecC2C(plan, freq.p, freq.p, CUFFT_FORWARD));
    spectrum<<<ceildiv(nt, 256), 256>>>(freq.p, response.p, k, nt);
    checked(cudaGetLastError());
    checked(cufftExecC2C(plan, freq.p, freq.p, CUFFT_INVERSE));
    unpack<<<ceildiv(nout, 256), 256>>>(freq.p, detector.p, g, k, nout, views.p + first);
    checked(cudaGetLastError());
    pairs<<<ceildiv(nout, 256), 256>>>(detector.p, freq.p, g.u, nout);
    checked(cudaGetLastError());
}
} // namespace fdk
