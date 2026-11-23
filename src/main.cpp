#include "files.hpp"
#include "finite.hpp"
#include "gpu.hpp"
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <map>
#include <memory>
#include <set>
#include <sstream>
using Clock = std::chrono::steady_clock;
struct Geom {
    int a, v, u, n;
    float du, dv, R, D, vox, start, sign;
};
static_assert(sizeof(void *) == 8 && sizeof(float) == 4, "Requires Windows x64 and FP32");
static_assert(sizeof(Geom) == 44, "kernel ABI");
int up(int x, int y) {
    return (x + y - 1) / y;
}
int integer(const std::string &s, int lo, int hi) {
    size_t pos;
    int n = std::stoi(s, &pos);
    if (pos != s.size() || n < lo || n > hi)
        throw std::runtime_error("out of range integer: " + s);
    return n;
}
int main() {
    try {
        auto argv = arguments();
        int argc = int(argv.size());
        if (argc == 2 && std::string(argv[1]) == "--help") {
            std::cout << "cone-fdk --geom scan.geom --input projections.f32 --output volume.f32 "
                         "--n 256 --fov 40 --budget 768 --batch 16 --ptx kernels.ptx "
                         "[--host-memory auto|pageable]\nBudget in "
                         "MiB; fov in mm. Metrics to stdout as JSON.\n";
            return 0;
        }
        auto begin = Clock::now();
        std::map<std::string, std::string> options;
        std::set<std::string> keys = {"--geom", "--input",  "--output", "--n",
                                      "--fov",  "--budget", "--batch",  "--ptx"};
        for (int i = 1; i < argc; i += 2) {
            if (i + 1 >= argc || (!keys.count(argv[i]) && argv[i] != "--host-memory") ||
                options.count(argv[i]))
                throw std::runtime_error("unknown/duplicate/incomplete argument");
            options[argv[i]] = argv[i + 1];
        }
        auto mode = options.count("--host-memory") ? options.at("--host-memory") : "auto";
        if (mode != "auto" && mode != "pageable")
            throw std::runtime_error("--host-memory expects auto or pageable");
        if (options.size() - options.count("--host-memory") != keys.size())
            throw std::runtime_error("missing argument; see --help");
        auto val = [&](const char *k) { return options.at(k); };
        auto dest = absolute(val("--output"));
        for (auto k : {"--input", "--geom", "--ptx"})
            if (same_file(dest, absolute(val(k))))
                throw std::runtime_error("output aliases an input");
        Geom g{};
        std::istringstream meta(small_file(val("--geom"), 4096));
        std::string magic, junk;
        if (!(meta >> magic >> g.a >> g.v >> g.u >> g.du >> g.dv >> g.R >> g.D >> g.start >>
              g.sign) ||
            magic != "CFDK1" || (meta >> junk))
            throw std::runtime_error("invalid CFDK1 geometry");
        g.n = integer(val("--n"), 2, 4096);
        size_t pos;
        float fov = std::stof(val("--fov"), &pos);
        if (pos != val("--fov").size())
            throw std::runtime_error("invalid fov");
        g.vox = fov / g.n;
        if (!(g.a >= 3 && g.a <= 65535 && g.u >= 2 && g.u <= 4096 && g.v >= 2 && g.v <= 4096 &&
              g.du > 0 && g.dv > 0 && g.D > g.R && g.R > fov * .708f && g.vox > 0 &&
              std::isfinite(g.D + g.R + g.du + g.dv + g.start + fov) && std::abs(g.sign) == 1))
            throw std::runtime_error("unsupported field of view or geometry");
        size_t budget = size_t(integer(val("--budget"), 1, 65536)) << 20;
        int batch = std::min(g.a, integer(val("--batch"), 1, 256));
        int k = 1;
        while (k < 2 * g.u)
            k *= 2;
        size_t row = size_t(g.u) * g.v, plane = size_t(g.n) * g.n;
        File in(absolute(val("--input")));
        if (in.size() != row * g.a * 4)
            throw std::runtime_error("projection file size mismatch");
        auto ptx = small_file(val("--ptx"), 16u << 20);
        Gpu gpu(ptx.c_str());
        size_t free0, total;
        CK(gpu.cuMemGetInfo_v2(&free0, &total));
        size_t guard = 128ull << 20;
        if (free0 <= guard)
            throw std::runtime_error("GPU is busy: insufficient free device memory");
        size_t cap = std::min(budget, free0 - guard);
        auto pre = gpu.kernel("pre"), sp = gpu.kernel("spectrum"), back = gpu.kernel("back"),
             un = gpu.kernel("unpack"), ang = gpu.kernel("angles"), ramp = gpu.kernel("ramp"),
             pairs = gpu.kernel("pairs");
        std::unique_ptr<Plan> plan;
        int slab = 0;
        size_t bytes = 0;
        for (; batch >= 1; batch /= 2) {
            size_t rows = size_t(batch) * g.v,
                   base = rows * k * 8 + row * batch * 4 + k * 4 + g.a * 8 + 6 * 2 * Buf::pad;
            if (rows * k > 500000000 || base + plane * 4 > cap)
                continue;
            plan.reset(new Plan(gpu, k, (int)rows));
            bytes = base + plan->work;
            if (bytes + plane * 4 <= cap) {
                slab = (int)std::min<size_t>(g.n, (cap - bytes) / (plane * 4));
                break;
            }
            plan.reset();
        }
        if (!slab)
            throw std::runtime_error(
                "budget cannot fit one view, FFT work area and one volume plane");
        unsigned long long elems = plane * slab;
        Buf block(gpu, elems * 4), views(gpu, row * batch * 4),
            freq(gpu, size_t(batch) * g.v * k * 8), cs(gpu, g.a * 8), h(gpu, k * 4),
            work(gpu, plan->work);
        CK(gpu.cufftSetWorkArea(plan->id, (void *)work.p));
        gpu.launch(ang, up(g.a, 128), 1, 1, 128, 1, cs.p, g);
        gpu.launch(ramp, up(k, 128), 1, 1, 128, 1, h.p, k);
        CK(gpu.cuCtxSynchronize());
        size_t free1;
        CK(gpu.cuMemGetInfo_v2(&free1, &total));
        HostBuffer host(gpu, row * batch, mode == "auto"),
            outChunk(gpu, std::min<size_t>(elems, 2u << 20), mode == "auto");
        Output output(dest);
        Timer timer(gpu);
        double filter = 0, bp = 0, h2d = 0, d2h = 0;
        int launches = 0;
        for (int z = 0; z < g.n; z += slab) {
            int depth = std::min(g.n - z, slab);
            CK(gpu.cuMemsetD32_v2(block.p, 0, plane * depth));
            in.rewind();
            for (int first = 0; first < g.a; first += batch) {
                int actual = std::min(batch, g.a - first), rows = batch * g.v, nt = rows * k,
                    nout = actual * g.v * g.u;
                size_t size = row * actual * 4;
                in.transfer(host.data(), size, false);
                if (!finite_block(host.data(), size / 4))
                    throw std::runtime_error("non-finite line integral");
                timer.start();
                CK(gpu.cuMemcpyHtoD_v2(views.p, host.data(), size));
                h2d += timer.stop();
                timer.start();
                gpu.launch(pre, up(nt, 256), 1, 1, 256, 1, views.p, freq.p, g, k, actual, rows);
                CK(gpu.cufftExecC2C(plan->id, (void *)freq.p, (void *)freq.p, -1));
                gpu.launch(sp, up(nt, 256), 1, 1, 256, 1, freq.p, h.p, k, nt);
                CK(gpu.cufftExecC2C(plan->id, (void *)freq.p, (void *)freq.p, 1));
                gpu.launch(un, up(nout, 256), 1, 1, 256, 1, freq.p, views.p, g, k, nout);
                // (u+1)*v+2 <= k*v because k>=2*u and u,v>=2.
                // nt is already bounded by the checked FFT planner.
                int entries = (g.u + 1) * g.v + 2, totalPairs = actual * entries;
                gpu.launch(pairs, up(totalPairs, 256), 1, 1, 256, 1, views.p, freq.p, g.u, g.v,
                           entries, totalPairs);
                filter += timer.stop();
                timer.start();
                gpu.launch(back, up(g.n, 16), up(g.n, 8), up(depth, 16), 16, 8, block.p, freq.p,
                           cs.p, g, first, actual, z, depth);
                bp += timer.stop();
                launches++;
            }
            for (size_t off = 0; off < plane * depth; off += outChunk.count) {
                size_t bytesOut = std::min(outChunk.count, plane * depth - off) * 4;
                timer.start();
                CK(gpu.cuMemcpyDtoH_v2(outChunk.data(), block.p + 4 * off, bytesOut));
                d2h += timer.stop();
                if (!finite_block(outChunk.data(), bytesOut / 4))
                    throw std::runtime_error("non-finite reconstructed attenuation");
                output.file->transfer(outChunk.data(), bytesOut, true);
            }
        }
        for (Buf *b : {&block, &views, &freq, &cs, &h, &work})
            b->verify();
        output.publish();
        auto wall = std::chrono::duration<double, std::milli>(Clock::now() - begin).count();
        std::cout << std::fixed << std::setprecision(4) << "{\"gpu\":\"" << gpu.name
                  << "\",\"driver_api\":" << gpu.driver << ",\"cufft_version\":" << gpu.fftver
                  << ",\"views\":" << g.a << ",\"n\":" << g.n << ",\"batch\":" << batch
                  << ",\"slab\":" << slab << ",\"slabs\":" << up(g.n, slab)
                  << ",\"launches_bp\":" << launches << ",\"budget_bytes\":" << budget
                  << ",\"redzones_verified\":true,\"peak_owned_bytes\":" << gpu.peak
                  << ",\"host_pinned_bytes\":" << host.pinnedBytes() + outChunk.pinnedBytes()
                  << ",\"fft_workspace_bytes\":" << plan->work << ",\"free_before_bytes\":" << free0
                  << ",\"free_at_allocation_bytes\":" << free1 << ",\"filter_ms\":" << filter
                  << ",\"bp_ms\":" << bp << ",\"h2d_ms\":" << h2d << ",\"d2h_ms\":" << d2h
                  << ",\"wall_ms\":" << wall << "}\n";
        return 0;
    } catch (const std::exception &e) {
        std::cerr << "cone-fdk: " << e.what() << "\n";
        return 2;
    }
}
