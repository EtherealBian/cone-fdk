// Local labels are slab-linear voxel indices. Parents strictly decrease until a root.
// Kernels execute in one stream; only the union phase has concurrent writers.
extern "C" __global__ void classify(const float *in, int *parent, int nx, int ny, int count,
                                    float threshold, int median) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count)
        return;
    float v = in[i];
    if (median) {
        int x = i % nx, y = (i / nx) % ny, base = i - y * nx - x;
        float a[9];
        int k = 0;
        for (int dy = -1; dy <= 1; ++dy)
            for (int dx = -1; dx <= 1; ++dx) {
                float t = in[base + max(0, min(ny - 1, y + dy)) * nx + max(0, min(nx - 1, x + dx))];
                int j = k++;
                while (j && a[j - 1] > t) {
                    a[j] = a[j - 1];
                    --j;
                }
                a[j] = t;
            }
        v = a[4];
    }
    parent[i] = v <= threshold ? i : -1;
}
__device__ int root(int *p, int a) {
    int b;
    while ((b = atomicAdd(p + a, 0)) != a) {
        int c = atomicAdd(p + b, 0);
        if (c != b)
            atomicCAS(p + a, b, c);
        a = c;
    }
    return a;
}
__device__ void edge(int *p, int i, int j) {
    if (atomicAdd(p + j, 0) < 0)
        return;
    for (;;) {
        int a = root(p, i), b = root(p, j);
        if (a == b)
            return;
        int hi = max(a, b), lo = min(a, b);
        if (atomicCAS(p + hi, hi, lo) == hi)
            return;
    }
}
extern "C" __global__ void connect(int *p, int nx, int ny, int count, int conn) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count || atomicAdd(p + i, 0) < 0)
        return;
    int x = i % nx, y = (i / nx) % ny, plane = nx * ny;
    if (x)
        edge(p, i, i - 1);
    if (conn == 6) {
        if (y)
            edge(p, i, i - nx);
        if (i >= plane)
            edge(p, i, i - plane);
        return;
    }
    if (y)
        for (int dx = -1; dx <= 1; ++dx)
            if (x + dx >= 0 && x + dx < nx)
                edge(p, i, i - nx + dx);
    if (i >= plane)
        for (int dy = -1; dy <= 1; ++dy)
            if (y + dy >= 0 && y + dy < ny)
                for (int dx = -1; dx <= 1; ++dx)
                    if (x + dx >= 0 && x + dx < nx)
                        edge(p, i, i - plane + dy * nx + dx);
}
extern "C" __global__ void resolve(const int *p, int *out, int count) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count)
        return;
    int a = p[i];
    if (a >= 0)
        while (p[a] != a)
            a = p[a];
    out[i] = a;
}
