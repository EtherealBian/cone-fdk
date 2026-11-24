// Seven donor masks are stored arm-major. Region zero is background.
extern "C" __global__ void votes(const unsigned char* masks, const int* region,
        unsigned char* codes, unsigned char* majority, unsigned int* hist, int n) {
    int v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v >= n) return;
    unsigned c = 0;
    for (unsigned a = 0; a < 7; ++a) c |= unsigned(masks[(unsigned long long)a*n+v] != 0) << a;
    codes[v] = (unsigned char)c;
    majority[v] = __popc(c) >= 4;
    if (region[v] > 0) atomicAdd(hist + (unsigned long long)region[v]*128 + c, 1u);
}
extern "C" __global__ void copy_winner(const unsigned char* codes, const int* region,
        const unsigned char* winners, unsigned char* out, int n) {
    int v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v < n) out[v] = (codes[v] >> winners[region[v]]) & 1;
}
