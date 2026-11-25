#pragma once
#include "gpu.hpp"
#include "wall_record.hpp"
#include <map>
#include <set>

constexpr const char *loss_usage =
    "cone-fdk loss --reference baseline.wall --reference-grid baseline.grid --input part.wall "
    "--cylinder part.grid --loss-mm MM --area-radius-mm MM --output regions.json --budget MiB "
    "--ptx kernels.ptx\n";

inline bool same_loss_grid(const CylinderGrid &a, const CylinderGrid &b) {
    const auto &x = a.c, &y = b.c;
    return a.shells == b.shells && x.w == y.w && x.h == y.h && x.pitch == y.pitch &&
           x.firstRadius == y.firstRadius && x.radialStep == y.radialStep &&
           std::equal(x.center, x.center + 3, y.center) && std::equal(x.axis, x.axis + 3, y.axis) &&
           std::equal(x.seam, x.seam + 3, y.seam);
}
struct LossRegion {
    int count = 0, firstRow = 0, lastRow = 0, peakAt = 0;
    bool seam = false, unknown = false;
    double sum = 0, peak = 0, minWall = std::numeric_limits<double>::infinity(), boreGrowth = 0,
           outerRecession = 0, peakRadius = 0;
};
inline int measure_loss(const std::vector<std::string> &args) {
    if (args.size() == 3 && args[2] == "--help") {
        std::cout << loss_usage;
        return 0;
    }
    const std::set<std::string> keys = {"--reference", "--reference-grid", "--input",
                                        "--cylinder",  "--loss-mm",        "--area-radius-mm",
                                        "--output",    "--budget",         "--ptx"};
    std::map<std::string, std::string> opt;
    for (size_t i = 2; i < args.size(); i += 2) {
        if (i + 1 >= args.size() || !keys.count(args[i]) || opt.count(args[i]))
            throw std::runtime_error(loss_usage);
        opt[args[i]] = args[i + 1];
    }
    if (opt.size() != keys.size())
        throw std::runtime_error(loss_usage);
    auto dest = absolute(opt.at("--output"));
    for (auto k : {"--reference", "--reference-grid", "--input", "--cylinder", "--ptx"})
        if (same_file(dest, absolute(opt.at(k))))
            throw std::runtime_error("loss output aliases input");
    auto grid = [&](const char *k) {
        return read_cylinder(opt.at(k), std::numeric_limits<double>::infinity());
    };
    auto g = grid("--cylinder");
    auto &c = g.c;
    if (!same_loss_grid(g, grid("--reference-grid")))
        throw std::runtime_error("cylinder grids differ; registration is not performed");
    auto number = [&](const char *k, auto &v) {
        std::istringstream s(opt.at(k));
        std::string extra;
        if (!plane_number(s, v) || (s >> extra))
            throw std::runtime_error(std::string("invalid ") + k);
    };
    double threshold, radius;
    int mib;
    number("--loss-mm", threshold);
    number("--area-radius-mm", radius);
    number("--budget", mib);
    const double last = c.firstRadius + (g.shells - 1) * c.radialStep;
    if (g.shells < 3 || !std::isfinite(threshold) || threshold <= 0 || !std::isfinite(radius) ||
        radius < c.firstRadius || radius > last || mib < 1 || mib > 65536)
        throw std::runtime_error("invalid loss threshold, datum radius, grid or budget");
    for (int j = 1; j < g.shells; ++j)
        if (!(c.firstRadius + j * c.radialStep > c.firstRadius + (j - 1) * c.radialStep))
            throw std::runtime_error("indistinct radial knots");
    const int w = c.w, h = c.h, n = w * h;
    const size_t mapBytes = size_t(n) * sizeof(WallSample), budget = size_t(mib) << 20;
    File ref(absolute(opt.at("--reference"))), part(absolute(opt.at("--input")));
    if (ref.size() != mapBytes || part.size() != mapBytes)
        throw std::runtime_error("wall map byte count mismatch");
    Gpu gpu(small_file(opt.at("--ptx"), 16u << 20).c_str());
    auto classify = gpu.kernel("loss_classify"), connect = gpu.kernel("loss_connect");
    size_t free, total;
    CK(gpu.cuMemGetInfo_v2(&free, &total));
    size_t guard = 128u << 20,
           cap = std::min<size_t>({budget, 16u << 20, free > guard ? free - guard : 0});
    size_t stride = size_t(w) * (2 * sizeof(WallSample) + sizeof(int));
    if (cap < 6 * Buf::pad + stride)
        throw std::runtime_error("loss budget cannot fit one complete row");
    int tile = int(std::min<size_t>(h, (cap - 6 * Buf::pad) / stride)), tn = tile * w;
    std::vector<WallSample> a(tn), b(tn);
    std::vector<int> labels(n);
    Buf da(gpu, tn * sizeof(WallSample)), db(gpu, tn * sizeof(WallSample)),
        dp(gpu, tn * sizeof(int));
    auto root = [&](int i) {
        while (labels[i] != i) {
            labels[i] = labels[labels[i]];
            i = labels[i];
        }
        return i;
    };
    for (int start = 0; start < h; start += tile) {
        int offset = start * w, m = std::min(tile, h - start) * w;
        size_t bytes = size_t(m) * sizeof(WallSample);
        ref.transfer(a.data(), bytes, false);
        part.transfer(b.data(), bytes, false);
        for (int i = 0; i < m; ++i)
            if (!valid_wall_record(a[i], c.firstRadius, last) ||
                !valid_wall_record(b[i], c.firstRadius, last))
                throw std::runtime_error("malformed wall record at index " +
                                         std::to_string(offset + i));
        CK(gpu.cuMemcpyHtoD_v2(da.p, a.data(), bytes));
        CK(gpu.cuMemcpyHtoD_v2(db.p, b.data(), bytes));
        gpu.launch(classify, (m + 127) / 128, 1, 1, 128, 1, da.p, db.p, dp.p, m, threshold);
        gpu.launch(connect, (m + 127) / 128, 1, 1, 128, 1, dp.p, w, m);
        CK(gpu.cuMemcpyDtoH_v2(labels.data() + offset, dp.p, size_t(m) * sizeof(int)));
        for (int j = 0; j < m; ++j) {
            int &p = labels[offset + j];
            if (p > j || p < -1 - int(WallStatusCount * WallStatusCount))
                throw std::runtime_error("invalid device label");
            if (p >= 0) {
                if (p != j && labels[offset + p] < 0)
                    throw std::runtime_error("broken device label");
                p = p == j ? offset + j : labels[offset + p];
            }
        }
        if (start)
            for (int col = 0; col < w; ++col) {
                int u = offset + col, v = u - w;
                if (labels[u] >= 0 && labels[v] >= 0) {
                    int x = root(u), y = root(v);
                    labels[std::max(x, y)] = std::min(x, y);
                }
            }
    }
    da.verify();
    db.verify();
    dp.verify();
    for (int i = 0; i < n; ++i)
        if (labels[i] >= 0)
            labels[i] = labels[labels[i]];
    const double pi = std::acos(-1.), cellArea = 2 * pi * radius / w * c.pitch;
    auto axial = [&](int row) { return (row - .5 * (h - 1)) * c.pitch; };
    Output out(dest);
    size_t written = 0;
    auto emit = [&](std::string s) {
        out.file->transfer(&s[0], s.size(), true);
        written += s.size();
    };
    std::ostringstream header;
    header << std::setprecision(17) << "{\"format\":\"CFDK-loss-v1\",\"rows\":" << h
           << ",\"columns\":" << w
           << ",\"connectivity\":4,\"periodic_columns\":true,\"threshold_mm\":" << threshold
           << ",\"area_radius_mm\":" << radius << ",\"cell_area_mm2\":" << cellArea
           << ",\"axial_pitch_mm\":" << c.pitch << ",\"center_mm\":";
    xyz(header, c.center, '[', ']');
    header << ",\"axis_unit\":";
    xyz(header, c.axis, '[', ']');
    header << ",\"seam_unit\":";
    xyz(header, c.seam, '[', ']');
    header << ",\"runs\":[\n";
    emit(header.str());
    // Files remain open read-only without write sharing. Re-read just the two row buffers
    // for statistics instead of retaining two complete FP64 maps in host memory.
    ref.rewind();
    part.rewind();
    std::map<int, LossRegion> regions;
    int hits = 0, unknown = 0;
    bool firstRun = true;
    for (int row = 0; row < h; ++row) {
        ref.transfer(a.data(), w * sizeof(WallSample), false);
        part.transfer(b.data(), w * sizeof(WallSample), false);
        std::ostringstream runs;
        for (int col = 0; col < w;) {
            const int start = col, id = labels[row * w + col];
            do {
                ++col;
            } while (col < w && labels[row * w + col] == id);
            if (id == -1)
                continue;
            runs << (firstRun ? "" : ",") << '[' << row << ',' << start << ',' << col - start << ','
                 << (id < 0 ? id : id + 1) << ']';
            firstRun = false;
            if (id < 0) {
                unknown += col - start;
                continue;
            }
            auto &r = regions[id];
            if (!r.count)
                r.firstRow = row;
            r.lastRow = row;
            hits += col - start;
            for (int x = start; x < col; ++x) {
                int i = row * w + x;
                const double d = a[x].thickness - b[x].thickness;
                if (!r.count || d > r.peak) {
                    r.peak = d;
                    r.peakAt = i;
                    r.peakRadius = b[x].outer;
                    r.boreGrowth = b[x].inner - a[x].inner;
                    r.outerRecession = a[x].outer - b[x].outer;
                }
                ++r.count;
                r.sum += d;
                r.minWall = std::min(r.minWall, b[x].thickness);
                int l = x ? i - 1 : i + w - 1, rt = x + 1 < w ? i + 1 : i - w + 1;
                r.seam |= x == 0 && labels[l] == id;
                r.unknown |= labels[l] < -1 || labels[rt] < -1 || (row && labels[i - w] < -1) ||
                             (row + 1 < h && labels[i + w] < -1);
            }
        }
        if (runs.tellp() > 0)
            emit(runs.str() + "\n");
    }
    std::ostringstream counts;
    counts << "],\"coverage\":{\"paired_valid\":" << n - unknown << ",\"unknown\":" << unknown
           << ",\"below_threshold\":" << n - unknown - hits << ",\"flagged\":" << hits
           << "},\"regions\":[\n";
    emit(counts.str());
    bool first = true;
    for (const auto &kv : regions) {
        const auto &r = kv.second;
        int y = r.peakAt / w, x = r.peakAt % w;
        double th = 2 * pi * x / w, world[3];
        for (int k = 0; k < 3; ++k)
            world[k] = c.center[k] + axial(y) * c.axis[k] +
                       r.peakRadius * (std::cos(th) * c.seam[k] + std::sin(th) * c.tangent[k]);
        std::ostringstream s;
        s << std::setprecision(17) << std::boolalpha << (first ? "" : ",")
          << "{\"id\":" << kv.first + 1 << ",\"cells\":" << r.count
          << ",\"area_mm2\":" << r.count * cellArea
          << ",\"axial_extent_mm\":" << (r.lastRow - r.firstRow + 1) * c.pitch
          << ",\"rows_inclusive\":[" << r.firstRow << ',' << r.lastRow
          << "],\"crosses_seam\":" << r.seam << ",\"touches_unknown\":" << r.unknown
          << ",\"touches_axial_edge\":" << (r.firstRow == 0 || r.lastRow == h - 1)
          << ",\"mean_loss_mm\":" << r.sum / r.count << ",\"min_remaining_mm\":" << r.minWall
          << ",\"peak\":{\"row\":" << y << ",\"column\":" << x
          << ",\"azimuth_deg\":" << 360. * x / w << ",\"axial_offset_mm\":" << axial(y)
          << ",\"loss_mm\":" << r.peak << ",\"inner_growth_mm\":" << r.boreGrowth
          << ",\"outer_recession_mm\":" << r.outerRecession << ",\"inspected_outer_xyz_mm\":";
        xyz(s, world, '[', ']');
        s << "}}\n";
        emit(s.str());
        first = false;
    }
    emit("]}\n");
    out.publish();
    std::cout << "{\"gpu\":\"" << gpu.name << "\",\"regions\":" << regions.size()
              << ",\"tiles\":" << (h + tile - 1) / tile << ",\"input_bytes_read\":" << 4 * mapBytes
              << ",\"output_bytes\":" << written << ",\"budget_bytes\":" << budget
              << ",\"peak_owned_bytes\":" << gpu.peak << ",\"redzones_verified\":true}\n";
    return 0;
}
