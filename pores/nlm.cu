// Patch-distance non-local means. The wide input tile supplies the 10-pixel
// search radius plus a 2-pixel patch halo without per-neighbour global loads.
extern "C" __global__ void nlm(const float *src, float *out, int w, int h, float bandwidth,
                               int weighted, float noise) {
    __shared__ float tile[40][40];
    const int tx = threadIdx.x, ty = threadIdx.y, sx = blockIdx.x * 16, sy = blockIdx.y * 16;
    const int z = blockIdx.z, base = z * w * h;
    for (int k = 16 * ty + tx; k < 1600; k += 256) {
        int y = k / 40, x = k % 40;
        tile[y][x] =
            src[base + min(max(sy + y - 12, 0), h - 1) * w + min(max(sx + x - 12, 0), w - 1)];
    }
    __syncthreads();
    if (sx + tx >= w || sy + ty >= h)
        return;
    float sum = 0, weights = 0;
    const float inv = 1.f / (25.f * bandwidth * bandwidth);
    for (int j = -10; j <= 10; j++)
        for (int i = -10; i <= 10; i++) {
            float d = 0;
#pragma unroll
            for (int v = -2; v <= 2; v++) {
#pragma unroll
                for (int u = -2; u <= 2; u++) {
                    float diff =
                        tile[ty + 12 + v][tx + 12 + u] - tile[ty + 12 + j + v][tx + 12 + i + u];
                    float wp = weighted ? (3 - abs(v)) * (3 - abs(u)) * (25.f / 81.f) : 1.f;
                    d = fmaf(diff * wp, diff, d);
                }
            }
            float a = __expf(-fmaxf(0.f, d - 25.f * noise * noise) * inv);
            sum = fmaf(a, tile[ty + 12 + j][tx + 12 + i], sum);
            weights += a;
        }
    out[base + (sy + ty) * w + sx + tx] = sum / weights;
}
