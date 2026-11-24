// Six-face membership from immutable labels; signed output has disjoint ownership.
// A background voxel adjacent to two different foreground IDs belongs to neither.
extern "C" __global__ void transfer_faces(const int* labels, const unsigned char* eligible,
                                          int* out, int nx, int ny, int nz) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    int plane = nx * ny;
    if (k >= plane * nz) return;
    int id = labels[k];
    out[k] = 0;
    if (id && !eligible[id]) return;
    int x = k % nx, y = (k / nx) % ny, z = k / plane;
    int near[6] = { x ? labels[k-1] : 0, x+1<nx ? labels[k+1] : 0,
                    y ? labels[k-nx] : 0, y+1<ny ? labels[k+nx] : 0,
                    z ? labels[k-plane] : 0, z+1<nz ? labels[k+plane] : 0 };
    int owner = 0;
    #pragma unroll
    for (int n : near) {
        if (id && n != id) { out[k] = -id; return; }
        if (!id && n) {
            if (owner && owner != n) return;
            owner = n;
        }
    }
    if (!id && eligible[owner]) out[k] = owner;
}
