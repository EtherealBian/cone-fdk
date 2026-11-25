#include "cylinder.hpp"
#include "files.hpp"
#include "finite.hpp"
#include "gpu.hpp"
#include "plane.hpp"
#include "regions.hpp"
#include "wall.hpp"
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <map>
#include <memory>
#include <set>
#include <sstream>
#include <utility>
using Clock = std::chrono::steady_clock;
static_assert(sizeof(void *) == 8 && sizeof(float) == 4, "Requires Windows x64 and FP32");
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
        if (argc >= 2 && argv[1] == "wall")
            return measure_wall(argv);
        if (argc == 2 && std::string(argv[1]) == "--help") {
            std::cout << "cone-fdk --geom scan.geom --input projections.f32 --output volume.f32 "
                         "--n 256 --fov 40 --budget 768 --batch 16 --ptx kernels.ptx "
                         "[--host-memory auto|pageable] [--roi x,y,z,nx,ny,nz | --roi-list "
                         "boxes.txt | --plane plane.txt | --stack stack.txt | --cylinder cyl.txt] "
                         "[--stack-format "
                         "raw|nrrd] [--planner "
                         "batch-first|min-passes]\nmin-passes requires "
                         "--roi-list, --stack or --cylinder. Budget in "
                         "MiB; fov in mm. Metrics to stdout as JSON.\n"
                      << wall_usage;
            return 0;
        }
        auto begin = Clock::now();
        std::map<std::string, std::string> options;
        std::set<std::string> keys = {"--geom", "--input",  "--output", "--n",
                                      "--fov",  "--budget", "--batch",  "--ptx"};
        for (int i = 1; i < argc; i += 2) {
            if (i + 1 >= argc ||
                (!keys.count(argv[i]) && argv[i] != "--host-memory" && argv[i] != "--roi" &&
                 argv[i] != "--roi-list" && argv[i] != "--planner" && argv[i] != "--plane" &&
                 argv[i] != "--cylinder" && argv[i] != "--stack" && argv[i] != "--stack-format") ||
                options.count(argv[i]))
                throw std::runtime_error("unknown/duplicate/incomplete argument");
            options[argv[i]] = argv[i + 1];
        }
        auto mode = options.count("--host-memory") ? options.at("--host-memory") : "auto";
        if (mode != "auto" && mode != "pageable")
            throw std::runtime_error("--host-memory expects auto or pageable");
        bool packed = options.count("--roi-list") != 0;
        auto policy = options.count("--planner") ? options.at("--planner") : "batch-first";
        if (policy != "batch-first" && policy != "min-passes")
            throw std::runtime_error("--planner expects batch-first or min-passes");
        bool stacked = options.count("--stack") != 0;
        bool cylindrical = options.count("--cylinder") != 0;
        auto format = options.count("--stack-format") ? options.at("--stack-format") : "raw";
        if ((options.count("--stack-format") && !stacked) || (format != "raw" && format != "nrrd"))
            throw std::runtime_error("--stack-format raw|nrrd requires --stack");
        bool nrrd = format == "nrrd";
        if (policy == "min-passes" && !packed && !stacked && !cylindrical)
            throw std::runtime_error(
                "--planner min-passes requires --roi-list, --stack or --cylinder");
        bool sliced = cylindrical || stacked || options.count("--plane");
        const char *spec = cylindrical ? "--cylinder" : stacked ? "--stack" : "--plane";
        if (int(cylindrical) + int(packed) + int(stacked) + int(options.count("--plane")) +
                int(options.count("--roi")) >
            1)
            throw std::runtime_error(
                "use only one of --roi, --roi-list, --plane, --stack and --cylinder");
        for (const auto &key : keys)
            if (!options.count(key))
                throw std::runtime_error("missing argument: " + key);
        auto val = [&](const char *k) { return options.at(k); };
        auto dest = absolute(val("--output"));
        for (auto k : {"--input", "--geom", "--ptx"})
            if (same_file(dest, absolute(val(k))))
                throw std::runtime_error("output aliases an input");
        if (packed && same_file(dest, absolute(val("--roi-list"))))
            throw std::runtime_error("output aliases the ROI list");
        if (sliced && same_file(dest, absolute(val(spec))))
            throw std::runtime_error("output aliases the grid specification");
        Geom g{};
        std::istringstream meta(small_file(val("--geom"), 4096));
        std::string magic, junk;
        if (!(meta >> magic >> g.a >> g.v >> g.u >> g.du >> g.dv >> g.R >> g.D >> g.start >>
              g.sign) ||
            magic != "CFDK1" || (meta >> junk))
            throw std::runtime_error("invalid CFDK1 geometry");
        g.n = integer(val("--n"), 2, 4096);
        bool cropped = packed || options.count("--roi");
        std::vector<Box> boxes;
        size_t pos;
        float fov = std::stof(val("--fov"), &pos);
        if (pos != val("--fov").size())
            throw std::runtime_error("invalid fov");
        g.vox = fov / g.n;
        if (!(g.a >= 3 && g.a <= 65535 && g.u >= 2 && g.u <= 4096 && g.v >= 2 && g.v <= 4096 &&
              g.du > 0 && g.dv > 0 && g.D > g.R && g.R > fov * .708f && g.vox > 0 &&
              std::isfinite(g.D + g.R + g.du + g.dv + g.start + fov) && std::abs(g.sign) == 1))
            throw std::runtime_error("unsupported field of view or geometry");
        PlaneGrid grid;
        CylinderGrid cyl;
        if (cylindrical) {
            cyl = read_cylinder(val(spec), fov);
            boxes.push_back({0, 0, 0, cyl.c.w, 1, cyl.c.h * cyl.shells});
        } else if (sliced) {
            grid = read_grid(val(spec), fov, stacked);
            // Flatten [layer,row] into one row stream: a pass can span a layer boundary.
            boxes.push_back({0, 0, 0, grid.p.w, 1, grid.p.h * grid.depth});
        } else if (packed)
            boxes = read_boxes(val("--roi-list"), g.n);
        else
            boxes.push_back(cropped ? parse_box(val("--roi"), g.n) : Box{0, 0, 0, g.n, g.n, g.n});
        size_t budget = size_t(integer(val("--budget"), 1, 65536)) << 20;
        const int requestedBatch = integer(val("--batch"), 1, 256);
        int k = 1;
        while (k < 2 * g.u)
            k *= 2;
        size_t row = size_t(g.u) * g.v, maxPlane = 0;
        for (const auto &b : boxes)
            maxPlane = std::max(maxPlane, b.plane());
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
        auto pre = gpu.kernel("pre"), sp = gpu.kernel("spectrum"),
             back = gpu.kernel(cylindrical ? "back_cylinder"
                               : sliced    ? "back_plane"
                               : cropped   ? "back_roi"
                                           : "back"),
             un = gpu.kernel("unpack"), ang = gpu.kernel("angles"), ramp = gpu.kernel("ramp"),
             pairs = gpu.kernel("pairs");
        std::unique_ptr<Plan> plan;
        // Score only feasible plans, with their actual cuFFT workspace. Sizing and
        // execution share the same ordered tiler, including whole-plane splits.
        struct Candidate {
            int batch;
            size_t passes;
        };
        std::vector<Candidate> candidates;
        std::vector<Box> tiles;
        size_t capacity = 0, elems = 0, bestPasses = 0;
        int batch = 0;
        for (int trial = std::min(g.a, requestedBatch); trial >= 1; trial /= 2) {
            size_t rows = size_t(trial) * g.v,
                   base = rows * k * 8 + row * trial * 4 + k * 4 + g.a * 8 + 6 * 2 * Buf::pad;
            if (rows * k > 500000000 || base + maxPlane * 4 > cap)
                continue;
            auto probe = std::make_unique<Plan>(gpu, k, int(rows));
            if (probe->work > cap - base - maxPlane * 4)
                continue;
            size_t room = (cap - base - probe->work) / 4;
            PassPlanner sizing(boxes, room);
            size_t peak = 0, count = 0;
            while (auto used = sizing.next(tiles)) {
                ++count;
                peak = std::max(peak, used);
            }
            candidates.push_back({trial, count});
            if (!plan || count < bestPasses) {
                plan = std::move(probe);
                batch = trial;
                capacity = room;
                elems = peak;
                bestPasses = count;
            }
            // Descending batch order settles ties; no plan can beat one pass.
            if (policy == "batch-first" || bestPasses == 1)
                break;
        }
        if (!plan)
            throw std::runtime_error(
                "budget cannot fit one view, FFT work area and one output row/volume plane");
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
        if (packed)
            write_pack_header(*output.file, boxes, g.n, g.vox);
        auto header = nrrd ? nrrd_header(grid) : std::string();
        if (nrrd)
            output.file->transfer(&header[0], header.size(), true);
        Timer timer(gpu);
        double filter = 0, bp = 0, h2d = 0, d2h = 0;
        uint64_t launches = 0, filterBatches = 0, passes = 0, inputBytes = 0;
        PassPlanner execution(boxes, capacity);
        while (auto used = execution.next(tiles)) {
            ++passes;
            CK(gpu.cuMemsetD32_v2(block.p, 0, used));
            in.rewind();
            for (int first = 0; first < g.a; first += batch) {
                int actual = std::min(batch, g.a - first), rows = batch * g.v, nt = rows * k,
                    nout = actual * g.v * g.u;
                size_t size = row * actual * 4;
                in.transfer(host.data(), size, false);
                inputBytes += size;
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
                ++filterBatches;
                Ptr dst = block.p;
                for (const auto &b : tiles) {
                    if (sliced) {
                        const int height = cylindrical ? cyl.c.h : grid.p.h;
                        for (int r = b.z; r < b.z + b.nz;) {
                            const int row0 = r % height;
                            const int rowsHere = std::min(height - row0, b.z + b.nz - r);
                            if (cylindrical) {
                                int shell = r / height;
                                gpu.launch(back, up(cyl.c.w, 16), up(rowsHere, 8), 1, 16, 8, dst,
                                           freq.p, cs.p, g, first, actual, cyl.c, shell, row0,
                                           rowsHere);
                            } else {
                                Plane plane = grid.at(r / height);
                                gpu.launch(back, up(plane.w, 16), up(rowsHere, 8), 1, 16, 8, dst,
                                           freq.p, cs.p, g, first, actual, plane, row0, rowsHere);
                            }
                            dst += size_t(rowsHere) * b.nx * 4;
                            r += rowsHere;
                            ++launches;
                        }
                        continue;
                    } else if (cropped) {
                        gpu.launch(back, up(b.nx, 16), up(b.ny, 8), up(b.nz + (b.z & 3), 16), 16, 8,
                                   dst, freq.p, cs.p, g, first, actual, b.z, b.nz, b.x, b.y, b.nx,
                                   b.ny);
                    } else {
                        gpu.launch(back, up(g.n, 16), up(g.n, 8), up(b.nz, 16), 16, 8, dst, freq.p,
                                   cs.p, g, first, actual, b.z, b.nz);
                    }
                    dst += b.voxels() * 4;
                    ++launches;
                }
                bp += timer.stop();
            }
            for (size_t off = 0; off < used; off += outChunk.count) {
                size_t bytesOut = std::min(outChunk.count, used - off) * 4;
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
        if (passes != bestPasses)
            throw std::runtime_error("internal error: planned pass count differs from execution");
        output.publish();
        auto wall = std::chrono::duration<double, std::milli>(Clock::now() - begin).count();
        std::cout << std::fixed << std::setprecision(4) << "{\"gpu\":\"" << gpu.name
                  << "\",\"driver_api\":" << gpu.driver << ",\"cufft_version\":" << gpu.fftver
                  << ",\"views\":" << g.a << ",\"n\":" << g.n << ",\"batch\":" << batch
                  << (sliced ? ",\"parent_voxel_spacing_mm\":" : ",\"voxel_spacing_mm\":")
                  << std::defaultfloat << std::setprecision(9) << g.vox << std::fixed
                  << std::setprecision(4);
        std::cout << ",\"planner\":\"" << policy << "\",\"requested_batch\":" << requestedBatch
                  << ",\"effective_budget_bytes\":" << cap << ",\"feasible_plans_scored\":[";
        for (size_t i = 0; i < candidates.size(); ++i) {
            if (i)
                std::cout << ",";
            std::cout << "{\"batch\":" << candidates[i].batch
                      << ",\"passes\":" << candidates[i].passes << "}";
        }
        std::cout << "]";
        size_t written = 0;
        if (sliced) {
            written = header.size() + 4 * boxes.front().voxels();
            if (cylindrical)
                print_cylinder(cyl);
            else
                print_grid(grid, stacked, nrrd, header.size());
            std::cout << ",\"rows_per_pass\":" << elems / boxes.front().nx;
        } else if (packed) {
            written = pack_header_bytes(boxes.size());
            std::cout << ",\"format\":\"CFDKROI1\",\"regions\":[";
            for (size_t i = 0; i < boxes.size(); ++i) {
                const auto &b = boxes[i];
                if (i)
                    std::cout << ",";
                std::cout << "{\"origin_xyz\":[" << b.x << "," << b.y << "," << b.z
                          << "],\"shape_xyz\":[" << b.nx << "," << b.ny << "," << b.nz
                          << "],\"offset_bytes\":" << written
                          << ",\"size_bytes\":" << 4 * b.voxels() << "}";
                written += 4 * b.voxels();
            }
            std::cout << "]";
        } else {
            const auto &b = boxes.front();
            written = 4 * b.voxels();
            std::cout << ",\"roi_origin_xyz\":[" << b.x << "," << b.y << "," << b.z
                      << "],\"output_shape_xyz\":[" << b.nx << "," << b.ny << "," << b.nz
                      << "],\"slab\":" << elems / b.plane() << ",\"slabs\":" << passes;
        }
        std::cout << ",\"output_bytes\":" << written << ",\"projection_passes\":" << passes
                  << ",\"projection_bytes_read\":" << inputBytes
                  << ",\"filter_batches\":" << filterBatches << ",\"launches_bp\":" << launches
                  << ",\"budget_bytes\":" << budget
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
