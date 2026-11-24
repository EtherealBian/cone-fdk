// Independent lattice-distance implementation, no CPU distance map or phase masks as input.
extern "C" __global__ void mass_terms(const unsigned short* raw, const unsigned char* foreign,
    const int* coords, unsigned long long* sums, int w, int h, int n, int nseed) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const int x = i % w, y = (i / w) % h, z = (i / w) / h;
    int best = 37;
    for (int j=0; j<nseed; ++j) {
        const int dz=z-coords[3*j], dy=y-coords[3*j+1], dx=x-coords[3*j+2];
        best = min(best, dz*dz + dy*dy + dx*dx);
    }
    if (best <= 36 && foreign[i]) atomicAdd(sums+4, 1ULL);
    if (best <= 9 || (best > 16 && best <= 36)) {
        const int k = best <= 9 ? 2 : 0;
        atomicAdd(sums+k, 1ULL);
        atomicAdd(sums+k+1, (unsigned long long)raw[i]);
    }
}
