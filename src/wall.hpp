#pragma once
#include "cylinder.hpp"
#include "gpu.hpp"
#include <map>

constexpr const char *wall_usage =
    "cone-fdk wall --cylinder grid.txt --input atlas.f32 --output radial.wall --threshold LEVEL "
    "--budget MiB --ptx kernels.ptx [--sector-cols ODD]\n";
inline int measure_wall(const std::vector<std::string> &args) {
    if (args.size() == 3 && args[2] == "--help") {
        std::cout << wall_usage;
        return 0;
    }
    std::map<std::string, std::string> o;
    for (size_t i = 2; i < args.size(); i += 2) {
        const auto &k = args[i];
        if (i + 1 >= args.size() ||
            (k != "--cylinder" && k != "--input" && k != "--output" && k != "--ptx" &&
             k != "--threshold" && k != "--budget" && k != "--sector-cols") ||
            o.count(k))
            throw std::runtime_error(wall_usage);
        o[k] = args[i + 1];
    }
    for (auto k : {"--input", "--output", "--ptx", "--cylinder", "--threshold", "--budget"})
        if (!o.count(k))
            throw std::runtime_error(wall_usage);
    auto dest = absolute(o.at("--output"));
    for (auto k : {"--input", "--ptx", "--cylinder"})
        if (same_file(dest, absolute(o.at(k))))
            throw std::runtime_error("wall output aliases an input");
    auto grid = read_cylinder(o.at("--cylinder"), std::numeric_limits<double>::infinity());
    const auto &c = grid.c;
    int d = grid.shells, w = c.w, h = c.h;
    if (d < 3)
        throw std::runtime_error("wall needs three or more distinct radial samples");
    for (int k = 1; k < d; ++k)
        if (!(c.firstRadius + k * c.radialStep > c.firstRadius + (k - 1) * c.radialStep))
            throw std::runtime_error("wall radial knots are not distinct in FP64");
    auto arg = [&](const char *key, auto &value) {
        std::istringstream ss(o.at(key));
        std::string extra;
        if (!plane_number(ss, value) || (ss >> extra))
            throw std::runtime_error(std::string("invalid ") + key);
    };
    int mib = 0, window = 2 * int(w / 48.) + 1;
    double level = 0;
    arg("--budget", mib);
    arg("--threshold", level);
    if (o.count("--sector-cols"))
        arg("--sector-cols", window);
    if (mib < 1 || mib > 65536 || !std::isfinite(level) || window < 1 || window > w ||
        window % 2 == 0)
        throw std::runtime_error("invalid wall budget, threshold or odd sector width");
    const int rays = h * w;
    File in(absolute(o.at("--input")));
    size_t inputBytes = size_t(rays) * d * sizeof(float);
    if (in.size() != inputBytes)
        throw std::runtime_error("wall atlas size mismatch");
    Gpu gpu(small_file(o.at("--ptx"), 16u << 20).c_str());
    auto kernel = gpu.kernel("radial_wall");
    size_t free, total;
    CK(gpu.cuMemGetInfo_v2(&free, &total));
    const size_t guard = 128u << 20, guards = 4 * Buf::pad;
    // Also bound host staging: a large requested device budget must not reserve gigabytes on the
    // host.
    const size_t budget = size_t(mib) << 20,
                 cap = std::min<size_t>({budget, 16u << 20, free > guard ? free - guard : 0});
    const size_t stride = d * sizeof(float) + sizeof(WallSample);
    if (cap < guards + stride)
        throw std::runtime_error("wall budget cannot fit one profile");
    int tile = int(std::min<size_t>(rays, (cap - guards) / stride));
    std::vector<float> host(size_t(tile) * d);
    std::vector<WallSample> result(tile);
    Buf atlas(gpu, host.size() * sizeof(float)), res(gpu, result.size() * sizeof(WallSample));
    Output output(dest);
    size_t counts[WallStatusCount] = {};
    std::vector<double> sums(w);
    std::vector<int> valid(w);
    double pointMin = std::numeric_limits<double>::infinity();
    int point = -1;
    for (int start = 0; start < rays; start += tile) {
        int n = std::min(tile, rays - start);
        for (int k = 0; k < d; ++k) {
            in.seek((size_t(k) * rays + start) * sizeof(float));
            in.transfer(host.data() + size_t(k) * n, n * sizeof(float), false);
        }
        CK(gpu.cuMemcpyHtoD_v2(atlas.p, host.data(), size_t(n) * d * sizeof(float)));
        gpu.launch(kernel, (n + 127) / 128, 1, 1, 128, 1, atlas.p, res.p, n, d, c.firstRadius,
                   c.radialStep, level);
        CK(gpu.cuMemcpyDtoH_v2(result.data(), res.p, n * sizeof(WallSample)));
        for (int j = 0; j < n; ++j) {
            const auto &s = result[j];
            if (s.status >= WallStatusCount || s.reserved)
                throw std::runtime_error("invalid GPU wall record");
            ++counts[s.status];
            if (s.status == Measured) {
                if (!(s.thickness > 0 && s.outer > s.inner && std::isfinite(s.outer)))
                    throw std::runtime_error("unrepresentable wall interfaces");
                int col = (start + j) % w;
                sums[col] += s.thickness;
                ++valid[col];
                if (s.thickness < pointMin) {
                    pointMin = s.thickness;
                    point = start + j;
                }
            }
        }
        output.file->transfer(result.data(), n * sizeof(WallSample), true);
    }
    atlas.verify();
    res.verify();
    // Circular windows may wrap over column zero. Never interpolate rejected profiles.
    int sector = -1, eligible = 0;
    double best = std::numeric_limits<double>::infinity();
    for (int col = 0; col < w; ++col) {
        double sum = 0;
        bool complete = true;
        for (int j = -window / 2; j <= window / 2; ++j) {
            int i = (col + j + w) % w;
            if (valid[i] != h) {
                complete = false;
                break;
            }
            sum += sums[i];
        }
        if (complete) {
            ++eligible;
            if (sum < best) {
                best = sum;
                sector = col;
            }
        }
    }
    output.publish();
    std::cout << std::setprecision(17) << "{\"format\":\"CFDK-wall-v1-le\",\"gpu\":\"" << gpu.name
              << "\",\"shape_ha\":[" << h << ',' << w << "],\"shells\":" << d
              << ",\"threshold\":" << level << ",\"first_radius_mm\":" << c.firstRadius
              << ",\"radial_step_mm\":" << c.radialStep << ",\"axial_pitch_mm\":" << c.pitch
              << ",\"center_mm\":";
    xyz(std::cout, c.center, '[', ']');
    std::cout << ",\"axis_unit\":";
    xyz(std::cout, c.axis, '[', ']');
    std::cout << ",\"seam_unit\":";
    xyz(std::cout, c.seam, '[', ']');
    const char *names[] = {"measured",  "empty",     "clipped",     "multiple",
                           "ambiguous", "nonfinite", "undersampled"};
    std::cout << ",\"status_counts\":{";
    for (unsigned i = 0; i < WallStatusCount; ++i)
        std::cout << (i ? "," : "") << '\"' << names[i] << "\":" << counts[i];
    std::cout << "},\"min_valid_point\":";
    if (point < 0)
        std::cout << "null";
    else
        std::cout << "{\"row\":" << point / w << ",\"column\":" << point % w
                  << ",\"mm\":" << pointMin << '}';
    std::cout << ",\"sector_cols\":" << window << ",\"eligible_sectors\":" << eligible
              << ",\"thinnest_complete_sector\":";
    if (sector < 0)
        std::cout << "null";
    else
        std::cout << "{\"column\":" << sector << ",\"azimuth_deg\":" << 360. * sector / w
                  << ",\"mean_mm\":" << best / (window * h) << '}';
    std::cout << ",\"input_bytes\":" << inputBytes
              << ",\"output_bytes\":" << size_t(rays) * sizeof(WallSample)
              << ",\"tiles\":" << (rays + tile - 1) / tile << ",\"budget_bytes\":" << budget
              << ",\"peak_owned_bytes\":" << gpu.peak << ",\"redzones_verified\":true}\n";
    return 0;
}
