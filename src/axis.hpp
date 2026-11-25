#pragma once
#include "wall_record.hpp"
#include "gpu.hpp"
#include <map>

constexpr const char *axis_usage =
    "cone-fdk axis --cylinder coarse.grid --input coarse.wall --output axis.json --max-residual-mm "
    "MM --budget MiB --ptx kernels.ptx\n";
inline int measure_axis(const std::vector<std::string> &args) {
    if (args.size() == 3 && args[2] == "--help") {
        std::cout << axis_usage;
        return 0;
    }
    std::map<std::string, std::string> o;
    for (size_t i = 2; i < args.size(); i += 2) {
        auto &k = args[i];
        if (i + 1 >= args.size() || o.count(k) ||
            (k != "--cylinder" && k != "--input" && k != "--output" && k != "--max-residual-mm" &&
             k != "--budget" && k != "--ptx"))
            throw std::runtime_error(axis_usage);
        o[k] = args[i + 1];
    }
    if (o.size() != 6)
        throw std::runtime_error(axis_usage);
    auto dest = absolute(o.at("--output"));
    for (auto k : {"--input", "--cylinder", "--ptx"})
        if (same_file(dest, absolute(o.at(k))))
            throw std::runtime_error("axis output aliases input");
    auto grid = read_cylinder(o.at("--cylinder"), std::numeric_limits<double>::infinity());
    auto &c = grid.c;
    int w = c.w, h = c.h, mib;
    double limit;
    std::string junk;
    std::istringstream mi(o.at("--budget")), mm(o.at("--max-residual-mm"));
    if (!plane_number(mi, mib) || mi >> junk || mib < 1 || mib > 65536 ||
        !plane_number(mm, limit) || mm >> junk || !std::isfinite(limit) || limit <= 0)
        throw std::runtime_error("invalid axis budget or residual limit");
    if (w % 2 || w < 32 || h < 3 || grid.shells < 3)
        throw std::runtime_error("axis requires even >=32 columns, >=3 rows and >=3 shells");
    File in(absolute(o.at("--input")));
    size_t row = size_t(w) * sizeof(WallSample), bytes = size_t(h) * row;
    if (in.size() != bytes)
        throw std::runtime_error("axis wall byte count mismatch");
    Gpu gpu(small_file(o.at("--ptx"), 16u << 20).c_str());
    auto sections = gpu.kernel("axis_sections"), distances = gpu.kernel("axis_distances");
    size_t free, total;
    CK(gpu.cuMemGetInfo_v2(&free, &total));
    size_t budget = size_t(mib) << 20, guard = 128u << 20,
           cap = std::min<size_t>({budget, 16u << 20, free > guard ? free - guard : 0}),
           stride = row + 3 * sizeof(double);
    if (cap < 4 * Buf::pad + stride)
        throw std::runtime_error("axis budget cannot fit one row");
    int tile = int(std::min<size_t>(h, (cap - 4 * Buf::pad) / stride));
    std::vector<WallSample> data(size_t(tile) * w);
    std::vector<double> result(3 * tile);
    Buf samples(gpu, data.size() * sizeof(WallSample)), fits(gpu, result.size() * sizeof(double));
    double x = 0, y = 0, kx = 0, ky = 0, zz = 0, sum = 0, lo = std::numeric_limits<double>::max(),
           hi = 0;
    int valid = 0, passes = 0, tiles = 0;
    bool singular = false;
    const char *why = nullptr;
    for (int pass = 0; pass < 2 && !why; ++pass) {
        in.rewind();
        ++passes;
        for (int start = 0; start < h; start += tile) {
            int n = std::min(tile, h - start);
            ++tiles;
            in.transfer(data.data(), n * row, false);
            if (!pass)
                for (int i = 0; i < n * w; ++i)
                    if (!valid_wall_record(data[i], c.firstRadius,
                                           c.firstRadius + (grid.shells - 1) * c.radialStep))
                        throw std::runtime_error("malformed axis wall at index " +
                                                 std::to_string(start * w + i));
            CK(gpu.cuMemcpyHtoD_v2(samples.p, data.data(), n * row));
            if (!pass)
                gpu.launch(sections, n, 1, 1, 128, 1, samples.p, fits.p, w);
            else
                gpu.launch(distances, n, 1, 1, 128, 1, samples.p, fits.p, w, h, start, c.pitch, x,
                           y, kx, ky);
            CK(gpu.cuMemcpyDtoH_v2(result.data(), fits.p, 3 * n * sizeof(double)));
            for (int j = 0; j < n; ++j) {
                const double *r = &result[3 * j];
                if (!pass) {
                    double z = (j + start - .5 * (h - 1)) * c.pitch;
                    valid += r[2] < 0 ? w : int(r[2]);
                    singular |= r[2] < 0 || !std::isfinite(r[0]) || !std::isfinite(r[1]);
                    x += r[0];
                    y += r[1];
                    kx += r[0] * z;
                    ky += r[1] * z;
                    zz += z * z;
                } else {
                    sum += r[0];
                    lo = std::min(lo, r[1]);
                    hi = std::max(hi, r[2]);
                }
            }
        }
        if (!pass) {
            if (valid != h * w)
                why = "incomplete";
            else if (singular || !(zz > 0))
                why = "ill_conditioned";
            else {
                x /= h;
                y /= h;
                kx /= zz;
                ky /= zz;
                if (!std::isfinite(kx) || !std::isfinite(ky))
                    why = "ill_conditioned";
                else if (std::hypot(kx, ky) > std::tan(20. * std::acos(-1.) / 180.))
                    why = "excess_tilt";
            }
        }
    }
    samples.verify();
    fits.verify();
    double radius = sum / (w * h), peak = std::max(hi - radius, radius - lo),
           span = (h - 1) * c.pitch;
    bool checked = passes == 2 && std::isfinite(radius) && radius > 0 && std::isfinite(peak);
    if (!why)
        why = !checked              ? "ill_conditioned"
              : span < .25 * radius ? "insufficient_span"
              : peak > limit        ? "noncylindrical"
                                    : nullptr;
    std::ostringstream s;
    s << std::setprecision(17) << "{\"format\":\"CFDK-axis-v1\",\"status\":\""
      << (why ? why : "measured") << "\",\"rows\":" << h << ",\"columns\":" << w
      << ",\"valid_samples\":" << valid << ",\"span_mm\":" << span
      << ",\"max_residual_mm\":" << limit << ",\"peak_residual_mm\":";
    if (checked)
        s << peak;
    else
        s << "null";
    s << ",\"measurement\":";
    if (why)
        s << "null";
    else {
        double norm = std::hypot(1., std::hypot(kx, ky)), center[3], axis[3], seam[3];
        for (int k = 0; k < 3; ++k) {
            center[k] = c.center[k] + x * c.seam[k] + y * c.tangent[k];
            axis[k] = (c.axis[k] + kx * c.seam[k] + ky * c.tangent[k]) / norm;
        }
        double dot = axis[0] * c.seam[0] + axis[1] * c.seam[1] + axis[2] * c.seam[2];
        for (int k = 0; k < 3; ++k)
            seam[k] = (c.seam[k] - dot * axis[k]) / std::sqrt(1 - dot * dot);
        s << "{\"origin_xyz_mm\":";
        xyz(s, center, '[', ']');
        s << ",\"axis_unit\":";
        xyz(s, axis, '[', ']');
        s << ",\"seam_unit\":";
        xyz(s, seam, '[', ']');
        s << ",\"outer_radius_mm\":" << radius << '}';
    }
    s << "}\n";
    auto payload = s.str();
    Output out(dest);
    out.file->transfer(&payload[0], payload.size(), true);
    out.publish();
    std::cout << "{\"gpu\":\"" << gpu.name << "\",\"passes\":" << passes << ",\"tiles\":" << tiles
              << ",\"input_bytes\":" << bytes << ",\"bytes_read\":" << bytes * passes
              << ",\"output_bytes\":" << payload.size() << ",\"budget_bytes\":" << budget
              << ",\"peak_owned_bytes\":" << gpu.peak << ",\"redzones_verified\":true}\n";
    return 0;
}
