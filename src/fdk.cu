#include "stages.cuh"
#include <algorithm>
#include <cmath>

static void validate_inputs(const Projections &proj, const Geometry &g) {
    if (!(g.a >= 3 && g.a <= 65535 && g.u >= 2 && g.u <= 4096 && g.v >= 2 && g.v <= 4096 &&
          g.n >= 2 && g.n <= 4096 && g.vox > 0 && g.du > 0 && g.dv > 0 && g.D > g.R &&
          g.R > g.n * g.vox * .708f && std::abs(g.sign) == 1 &&
          std::isfinite(g.vox * g.n + g.du + g.dv + g.R + g.D + g.start)) ||
        proj.size() != size_t(g.a) * g.v * g.u)
        throw std::invalid_argument("invalid geometry or projection extent");
}
void fdk_reconstruct(const Projections &proj, const Geometry &geo, Volume &out) {
    validate_inputs(proj, geo);
    const int batch = std::min(16, geo.a);
    const size_t pixels = size_t(geo.v) * geo.u, voxels = size_t(geo.n) * geo.n * geo.n;
    fdk::Stages stages(geo, batch);
    Device<float> volume(voxels);
    checked(cudaMemset(volume.p, 0, voxels * sizeof(float)));
    for (int first = 0; first < geo.a; first += batch) {
        int actual = std::min(batch, geo.a - first);
        checked(cudaMemcpy(stages.detector.p, proj.data() + first * pixels,
                           actual * pixels * sizeof(float), cudaMemcpyHostToDevice));
        stages.filter(actual);
        stages.backproject(volume.p, first, actual);
    }
    out.resize(voxels);
    checked(cudaMemcpy(out.data(), volume.p, voxels * sizeof(float), cudaMemcpyDeviceToHost));
}
