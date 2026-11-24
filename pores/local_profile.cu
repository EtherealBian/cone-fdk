// Numeric binary gradient-boosted trees, no framework on the inference path.
extern "C" __global__ void tree_score(const float *x, const int *feat,
    const int *left, const int *right, const double *split_or_value,
    const int *roots, double base, double *out, int n, int d, int trees) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=n) return;
    double s=base;
    for(int t=0;t<trees;++t){
        int j=roots[t];
        while(feat[j]>=0) j=(double)x[(long long)i*d+feat[j]]<=split_or_value[j]?left[j]:right[j];
        s+=split_or_value[j];
    }
    out[i]=s;
}
