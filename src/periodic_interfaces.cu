// Two-bit facets in increasing significance: north, east, south, west.
// 3 is an axial exterior sentinel inserted by the host, never a measured input value.
__device__ unsigned face(unsigned char neighbour) {
    return neighbour == 1 ? 0u : neighbour == 0 ? 1u : unsigned(neighbour);
}
extern "C" __global__ void encode_interfaces(const unsigned char *halo, unsigned char *out,
                                             int width, int cells) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= cells)
        return;
    const int p = i + width;
    if (halo[p] != 1) {
        out[i] = 0;
        return;
    }
    const int x = i % width;
    const int left = x == 0 ? p + width - 1 : p - 1;
    const int right = x + 1 == width ? p - x : p + 1;
    out[i] = static_cast<unsigned char>(face(halo[p - width]) | (face(halo[right]) << 2) |
                                        (face(halo[p + width]) << 4) | (face(halo[left]) << 6));
}
