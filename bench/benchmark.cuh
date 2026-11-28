#pragma once
#include "stages.cuh"
#include <algorithm>
#include <cstdio>
#include <exception>
#include <string>
#include <vector>
// One angle batch. Uploads, volume clears and warmup are outside event timing.
template <bool Filter> int benchmark(int argc, char **argv) try {
    if (argc != 4)
        throw std::invalid_argument("usage: bench_STAGE grid_n views repeats");
    int n = std::stoi(argv[1]), a = std::stoi(argv[2]), reps = std::stoi(argv[3]);
    if (n < 2 || n > 1024 || a < 3 || a > 256 || reps < 1)
        throw std::invalid_argument("n=2..1024, views=3..256, repeats>=1");
    Geometry g{a, n, n, n, 1, 1, 2.f * n, 3.f * n, 1, 0, 1};
    fdk::Stages stages(g, a);
    size_t plane = size_t(n) * n, pixels = plane * a, voxels = plane * n;
    std::vector<float> host(pixels);
    for (size_t i = 0; i < pixels; ++i)
        host[i] = float(i % 251) / 251.f;
    Device<float> out(Filter ? 1 : voxels);
    Events ev;
    auto upload = [&] {
        checked(cudaMemcpy(stages.detector.p, host.data(), pixels * 4, cudaMemcpyHostToDevice));
    };
    if constexpr (!Filter) {
        upload();
        stages.filter(a);
    }
    std::vector<float> times;
    for (long long i = 0; i < reps + 1LL; ++i) {
        if constexpr (Filter)
            upload();
        else
            CUDA_CHECK(cudaMemset(out.p, 0, voxels * 4));
        ev.begin();
        if constexpr (Filter)
            stages.filter(a);
        else
            stages.backproject(out.p, 0, a);
        const float ms = ev.finish();
        if (i)
            times.push_back(ms);
    }
    std::sort(times.begin(), times.end());
    double t = (times[(reps - 1) / 2] + times[reps / 2]) * .5;
    size_t bytes = Filter ? pixels * 4 + size_t(a) * stages.entries * 8 : voxels * 8;
    std::printf("| %s | %.5f | n/a | %.3f GB/s (effective) | n/a |\n",
                Filter ? "Filter (cuFFT)" : "Backprojection", t, bytes / (t * 1e6));
    return 0;
} catch (const std::exception &e) {
    std::fprintf(stderr, "%s\n", e.what());
    return 1;
}
