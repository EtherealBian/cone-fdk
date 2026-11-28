#include "stages.cuh"
/**
 * angles — Precompute one sine/cosine pair per view.
 *
 * Thread mapping: One thread per view; independent evaluations.
 * Memory: Global float2 table; reused across angle batches.
 * Occupancy: 8.23% achieved; 18 registers/thread (GB10, first launch).
 * Known bottleneck: Setup only; see optimization-log, geometry table.
 */
extern "C" __global__ void angles(float2 *__restrict__ cs, Geometry g) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= g.a)
        return;
    float c, s;
    sincosf(g.start + g.sign * 6.283185307179586f * i / g.a, &s, &c);
    cs[i] = make_float2(c, s);
}
// DFT of the finite, symmetric spatial Ram-Lak impulse. Nonzero discrete DC
// matters on small detectors; |f| sampled directly is not the same filter.
/**
 * ramp — Construct the discrete Ram-Lak frequency response.
 *
 * Thread mapping: One thread per frequency bin.
 * Memory: One global output per bin; inner sum in FP64.
 * Occupancy: 8.08% achieved; 36 registers/thread (GB10, first launch).
 * Known bottleneck: Setup only; see optimization-log, discrete filter.
 */
extern "C" __global__ void ramp(float *__restrict__ dst, int k) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= k)
        return;
    double h = .25;
    for (int j = 1; j < k / 2; j += 2)
        h -= 2. * cos(6.2831853071795864769 * i * j / k) / (9.869604401089358619 * j * j);
    dst[i] = (float)(h / k);
}
/**
 * pre — Cone-weight detector rows and pad the FFT batch.
 *
 * Thread mapping: One thread per padded detector element.
 * Memory: Contiguous complex writes; valid detector columns read FP32.
 * Occupancy: 77.59% achieved; 18 registers/thread (GB10, first launch).
 * Known bottleneck: Padding adds traffic; see optimization-log, batched filter.
 */
extern "C" __global__ void pre(const float *__restrict__ in, float2 *__restrict__ f, Geometry g,
                               int k, int actual, int rows) {
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
/**
 * spectrum — Scale each FFT bin by the discrete response.
 *
 * Thread mapping: One thread per complex frequency bin.
 * Memory: In-place global complex array, shared read-only response across rows.
 * Occupancy: 82.93% achieved; 16 registers/thread (GB10, first launch).
 * Known bottleneck: Bandwidth candidate; see optimization-log, batched filter.
 */
extern "C" __global__ void spectrum(float2 *__restrict__ f, const float *__restrict__ h, int k,
                                    int total) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= total)
        return;
    float w = h[t % k];
    f[t].x *= w;
    f[t].y *= w;
}
/**
 * unpack — Normalize the inverse transform into the detector buffer.
 *
 * Thread mapping: One thread per real detector sample.
 * Memory: Read a padded complex row and write the unpadded real row.
 * Occupancy: 81.71% achieved; 17 registers/thread (GB10, first launch).
 * Known bottleneck: See optimization-log, batched filter.
 */
extern "C" __global__ void unpack(const float2 *__restrict__ f, float *__restrict__ out, Geometry g,
                                  int k, int total) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= total)
        return;
    out[t] = f[(t / g.u) * k + t % g.u].x * (3.141592653589793f / g.a) * (g.D / (g.R * g.du));
}
// Pack horizontal interpolation into (left, right-left) once per detector sample,
// not once per voxel. Reuse the now-dead complex FFT allocation. Each view
// has a virtual left column and two zero guard entries for vertical clipping.
/**
 * pairs — Prepare horizontal interpolation coefficients.
 *
 * Thread mapping: One thread per LUT entry, including border sentinels.
 * Memory: Global float2 pairs overwrite the now-dead FFT scratch.
 * Occupancy: 77.76% achieved; 24 registers/thread (GB10, first launch).
 * Known bottleneck: Extra pass trades filter traffic for fewer repeated subtractions; see
 * optimization-log.
 */
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

namespace fdk {
static int fft_width(int u) {
    int k = 1;
    while (k < 2 * u)
        k *= 2;
    return k;
}
Stages::Stages(Geometry geom, int b)
    : g(geom), k(fft_width(g.u)), rows(b * g.v), entries((g.u + 1) * g.v + 2),
      detector(size_t(rows) * g.u), response(k), freq(size_t(rows) * k), cs(g.a) {
    angles<<<ceildiv(g.a, 128), 128>>>(cs.p, g);
    CUDA_CHECK_LAST();
    ramp<<<ceildiv(k, 128), 128>>>(response.p, k);
    CUDA_CHECK_LAST();
    fft_checked(cufftPlan1d(&plan, k, CUFFT_C2C, rows));
}
void Stages::filter(int actual) {
    const int nt = rows * k, nout = actual * g.v * g.u, np = actual * entries;
    pre<<<ceildiv(nt, 256), 256>>>(detector.p, freq.p, g, k, actual, rows);
    CUDA_CHECK_LAST();
    fft_checked(cufftExecC2C(plan, freq.p, freq.p, CUFFT_FORWARD));
    spectrum<<<ceildiv(nt, 256), 256>>>(freq.p, response.p, k, nt);
    CUDA_CHECK_LAST();
    fft_checked(cufftExecC2C(plan, freq.p, freq.p, CUFFT_INVERSE));
    unpack<<<ceildiv(nout, 256), 256>>>(freq.p, detector.p, g, k, nout);
    CUDA_CHECK_LAST();
    pairs<<<ceildiv(np, 256), 256>>>(detector.p, freq.p, g.u, g.v, entries, np);
    CUDA_CHECK_LAST();
}
} // namespace fdk
