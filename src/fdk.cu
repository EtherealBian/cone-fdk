#include "stages.cuh"
#include <algorithm>
#include <cmath>
#include <numeric>

namespace {
struct Event {
    cudaEvent_t handle;
    Event() {
        checked(cudaEventCreate(&handle));
    }
    ~Event() {
        cudaEventDestroy(handle);
    }
    Event(const Event &) = delete;
    void record() {
        checked(cudaEventRecord(handle));
    }
};

std::vector<fdk::View> make_views(const std::vector<Pose> &poses) {
    constexpr double pi = 3.14159265358979323846;
    const int n = static_cast<int>(poses.size());
    std::vector<int> order(n);
    std::iota(order.begin(), order.end(), 0);
    std::stable_sort(order.begin(), order.end(),
                     [&](int i, int j) { return poses[i].degrees < poses[j].degrees; });
    std::vector<fdk::View> views(n);
    for (int k = 0; k < n; ++k) {
        const int i = order[k];
        auto p = poses[i];
        double gap = poses[order[(k + 1) % n]].degrees - poses[order[(k + n - 1) % n]].degrees;
        if (k == 0 || k == n - 1)
            gap += 360.;
        const double angle = p.degrees * pi / 180.;
        // Circular trapezoid quadrature, then 1/2 for a full rotation.
        views[i] = {float(std::cos(angle)), float(std::sin(angle)), float(p.offset_x),
                    float(p.offset_y), float(gap * pi / 720.)};
    }
    return views;
}
} // namespace

float fdk_reconstruct(const Projections &proj, const Geometry &g, const std::vector<Pose> &poses,
                      Volume &out) {
    const int batch = std::min(16, g.a), batches = fdk::ceildiv(g.a, batch);
    const size_t pixels = size_t(g.v) * g.u, voxels = size_t(g.n) * g.n * g.n;
    fdk::Stages stages(g, batch, make_views(poses));
    Device<float> volume(voxels);
    checked(cudaMemset(volume.p, 0, voxels * sizeof(float)));
    std::vector<Event> start(batches), stop(batches);
    for (int first = 0, b = 0; first < g.a; first += batch, ++b) {
        const int actual = std::min(batch, g.a - first);
        checked(cudaMemcpy(stages.detector.p, proj.data() + first * pixels,
                           actual * pixels * sizeof(float), cudaMemcpyHostToDevice));
        stages.filter(first, actual);
        start[b].record();
        stages.backproject(volume.p, first, actual);
        stop[b].record();
    }
    out.resize(voxels);
    checked(cudaMemcpy(out.data(), volume.p, voxels * sizeof(float), cudaMemcpyDeviceToHost));
    float sum = 0;
    for (int b = 0; b < batches; ++b) {
        float ms;
        checked(cudaEventElapsedTime(&ms, start[b].handle, stop[b].handle));
        sum += ms;
    }
    return sum;
}
