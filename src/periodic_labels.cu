// Exact four-connected components on one material cylinder. No axial wrap.
// Every live parent is an index in this plane and can only decrease. CAS on a
// still-root node is the linearization point of a union; stale roots are retried.
extern "C" __global__ void init_forest(int *parent, int n) {
    int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p < n)
        parent[p] = p;
}
__device__ int climb(int *parent, int i) {
    for (;;) {
        int p = atomicAdd(parent + i, 0);
        if (p == i)
            return i;
        int gp = atomicAdd(parent + p, 0);
        if (gp != p)
            atomicCAS(parent + i, p, gp);
        i = gp;
    }
}
__device__ void connect(int *parent, int a, int b) {
    for (;;) {
        a = climb(parent, a);
        b = climb(parent, b);
        if (a == b)
            return;
        int hi = a > b ? a : b, lo = a > b ? b : a;
        if (atomicCAS(parent + hi, hi, lo) == hi)
            return;
    }
}
extern "C" __global__ void link_cylinder(const unsigned char *mask, int *parent, int n, int w) {
    int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= n || mask[p] != 1)
        return;
    int left = (p % w) ? p - 1 : p + w - 1;
    if (mask[left] == 1)
        connect(parent, p, left);
    if (p >= w && mask[p - w] == 1)
        connect(parent, p, p - w);
}
// The prior launch has finished. No thread mutates the forest during emission.
extern "C" __global__ void emit_canonical(const unsigned char *mask, const int *parent, int *label,
                                          int n) {
    int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= n)
        return;
    if (mask[p] != 1) {
        label[p] = mask[p] == 2 ? -1 : 0;
        return;
    }
    int root = p;
    while (parent[root] != root)
        root = parent[root];
    label[p] = root + 1;
}
