// Two patch sizes in one pass. The 21x21 search is shared; pixel intensities are
// averaged independently using 5x5 and 7x7 patch distances at three bandwidths.
extern "C" __global__ void multiscale(const float *src, float *out, int w, int h, float bandwidth,
                                      int, float) {
    __shared__ float tile[42][42];
    int tx = threadIdx.x, ty = threadIdx.y, x = blockIdx.x * 16 + tx, y = blockIdx.y * 16 + ty,
        z = blockIdx.z;
    for (int k = ty * 16 + tx; k < 42 * 42; k += 256) {
        int ix = blockIdx.x * 16 + k % 42 - 13, iy = blockIdx.y * 16 + k / 42 - 13;
        tile[k / 42][k % 42] = src[(z * h + max(0, min(h - 1, iy))) * w + max(0, min(w - 1, ix))];
    }
    __syncthreads();
    if (x >= w || y >= h)
        return;
    const float inv = 1.f / (bandwidth * bandwidth);
    float sw[6] = {}, sv[6] = {};
    for (int j = -10; j <= 10; j++)
        for (int i = -10; i <= 10; i++) {
            float inner = 0, outer = 0;
#pragma unroll
            for (int v = -3; v <= 3; v++) {
#pragma unroll
                for (int u = -3; u <= 3; u++) {
                    float d =
                        tile[ty + 13 + v][tx + 13 + u] - tile[ty + 13 + j + v][tx + 13 + i + u];
                    if (abs(u) <= 2 && abs(v) <= 2)
                        inner = fmaf(d, d, inner);
                    else
                        outer = fmaf(d, d, outer);
                }
            }
            float pix = tile[ty + 13 + j][tx + 13 + i];
#pragma unroll
            for (int k = 0; k < 6; k++) {
                float d = k < 3 ? inner * (1.f / 25.f) : (inner + outer) * (1.f / 49.f);
                float factor = k % 3 == 0 ? (1.f / .49f) : (k % 3 == 1 ? 1.f : (1.f / 1.69f));
                float a = __expf(-d * inv * factor);
                sw[k] += a;
                sv[k] = fmaf(a, pix, sv[k]);
            }
        }
#pragma unroll
    for (int k = 0; k < 6; k++)
        out[((z * 6 + k) * h + y) * w + x] = sv[k] / sw[k];
}
