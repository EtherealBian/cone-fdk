#pragma once
#include "wall_record.hpp"
#include "gpu.hpp"
#include <map>

constexpr const char *bore_usage =
    "cone-fdk bore --cylinder grid.txt --input radial.wall --output sections.json "
    "--max-residual-mm MM --budget MiB --ptx kernels.ptx\n";

inline bool bore_solved(const BoreCircle &f) {
    return f.solved == 1 && std::isfinite(f.x) && std::isfinite(f.y) && std::isfinite(f.radius) &&
           f.radius > 0 && std::isfinite(f.rms) && std::isfinite(f.peak);
}
inline void bore_world_circle(std::ostream &ss, const BoreCircle &f, const Cylinder &c, double z) {
    double xyzmm[3];
    for (int k = 0; k < 3; ++k)
        xyzmm[k] = c.center[k] + z * c.axis[k] + f.x * c.seam[k] + f.y * c.tangent[k];
    ss << "{\"diameter_mm\":" << 2 * f.radius << ",\"center_xyz_mm\":";
    xyz(ss, xyzmm, '[', ']');
    ss << '}';
}
inline int measure_bore(const std::vector<std::string> &args) {
    if (args.size() == 3 && args[2] == "--help") {
        std::cout << bore_usage;
        return 0;
    }
    std::map<std::string, std::string> o;
    for (size_t i = 2; i < args.size(); i += 2) {
        const auto &k = args[i];
        if (i + 1 >= args.size() ||
            (k != "--cylinder" && k != "--input" && k != "--output" && k != "--ptx" &&
             k != "--max-residual-mm" && k != "--budget") ||
            o.count(k))
            throw std::runtime_error(bore_usage);
        o[k] = args[i + 1];
    }
    if (o.size() != 6)
        throw std::runtime_error(bore_usage);
    auto dest = absolute(o.at("--output"));
    for (auto k : {"--cylinder", "--input", "--ptx"})
        if (same_file(dest, absolute(o.at(k))))
            throw std::runtime_error("bore output aliases an input");
    auto g = read_cylinder(o.at("--cylinder"), std::numeric_limits<double>::infinity());
    auto &c = g.c;
    int w = c.w, h = c.h;
    if (w < 32 || g.shells < 3)
        throw std::runtime_error("bore needs >=32 azimuths and a wall grid with >=3 radii");
    int mib;
    double limit;
    std::string junk;
    std::istringstream mi(o.at("--budget")), mm(o.at("--max-residual-mm"));
    if (!plane_number(mi, mib) || mi >> junk || mib < 1 || mib > 65536 ||
        !plane_number(mm, limit) || mm >> junk || !std::isfinite(limit) || limit <= 0)
        throw std::runtime_error("invalid bore budget or positive residual limit");
    File in(absolute(o.at("--input")));
    size_t rowBytes = size_t(w) * sizeof(WallSample);
    if (in.size() != size_t(h) * rowBytes)
        throw std::runtime_error("bore wall byte count mismatch");
    Gpu gpu(small_file(o.at("--ptx"), 16u << 20).c_str());
    auto kernel = gpu.kernel("bore_circles");
    size_t free, total;
    CK(gpu.cuMemGetInfo_v2(&free, &total));
    size_t budget = size_t(mib) << 20, guard = 128u << 20,
           cap = std::min<size_t>({budget, 16u << 20, free > guard ? free - guard : 0}),
           stride = rowBytes + 2 * sizeof(BoreCircle);
    if (cap < 4 * Buf::pad + stride)
        throw std::runtime_error("bore budget cannot fit one complete row");
    int tile = int(std::min<size_t>(h, (cap - 4 * Buf::pad) / stride));
    std::vector<WallSample> data(size_t(tile) * w);
    std::vector<BoreCircle> fits(tile * 2);
    Buf samples(gpu, data.size() * sizeof(WallSample)),
        pairs(gpu, fits.size() * sizeof(BoreCircle));
    Output output(dest);
    size_t bytes = 0;
    auto write = [&](std::string s) {
        output.file->transfer(&s[0], s.size(), true);
        bytes += s.size();
    };
    std::ostringstream head;
    head << std::setprecision(17) << "{\"format\":\"CFDK-bore-v1\",\"max_residual_mm\":" << limit
         << ",\"columns\":" << w << ",\"axis_unit\":";
    xyz(head, c.axis, '[', ']');
    head << ",\"seam_unit\":";
    xyz(head, c.seam, '[', ']');
    head << ",\"rows\":[\n";
    write(head.str());
    int accepted = 0;
    for (int start = 0; start < h; start += tile) {
        int n = std::min(tile, h - start);
        in.transfer(data.data(), n * rowBytes, false);
        for (int i = 0; i < n * w; ++i)
            if (!valid_wall_record(data[i], c.firstRadius,
                                   c.firstRadius + (g.shells - 1) * c.radialStep))
                throw std::runtime_error("malformed wall record at index " +
                                         std::to_string(start * w + i));
        CK(gpu.cuMemcpyHtoD_v2(samples.p, data.data(), n * rowBytes));
        gpu.launch(kernel, 2 * n, 1, 1, 128, 1, samples.p, pairs.p, w);
        CK(gpu.cuMemcpyDtoH_v2(fits.data(), pairs.p, 2 * n * sizeof(BoreCircle)));
        for (int j = 0; j < n; ++j) {
            auto &a = fits[2 * j], &b = fits[2 * j + 1];
            int row = start + j;
            if (a.valid != b.valid || a.valid < 0 || a.valid > w)
                throw std::runtime_error("invalid bore coverage");
            bool solved = bore_solved(a) && bore_solved(b);
            double ecc = std::hypot(a.x - b.x, a.y - b.y), gap = b.radius - a.radius - ecc,
                   z = (row - .5 * (h - 1)) * c.pitch;
            const char *why = a.valid < w                        ? "incomplete"
                              : !solved                          ? "ill_conditioned"
                              : std::max(a.peak, b.peak) > limit ? "noncircular"
                              : gap <= 0                         ? "not_nested"
                                                                 : nullptr;
            std::ostringstream s;
            s << std::setprecision(17) << (row ? "," : "") << "{\"row\":" << row
              << ",\"axial_offset_mm\":" << z << ",\"valid_columns\":" << a.valid
              << ",\"status\":\"" << (why ? why : "measured") << "\",\"residual_mm\":";
            if (solved)
                s << "{\"inner_rms\":" << a.rms << ",\"inner_peak\":" << a.peak
                  << ",\"outer_rms\":" << b.rms << ",\"outer_peak\":" << b.peak << '}';
            else
                s << "null";
            s << ",\"measurement\":";
            if (why)
                s << "null";
            else {
                ++accepted;
                s << "{\"inner\":";
                bore_world_circle(s, a, c, z);
                s << ",\"outer\":";
                bore_world_circle(s, b, c, z);
                s << ",\"eccentricity_mm\":" << ecc << ",\"model_min_gap_mm\":" << gap << "}";
            }
            s << "}\n";
            write(s.str());
        }
    }
    samples.verify();
    pairs.verify();
    write("]}\n");
    output.publish();
    std::cout << "{\"gpu\":\"" << gpu.name << "\",\"rows\":" << h << ",\"measured\":" << accepted
              << ",\"rejected\":" << h - accepted << ",\"tiles\":" << (h + tile - 1) / tile
              << ",\"input_bytes\":" << h * rowBytes << ",\"output_bytes\":" << bytes
              << ",\"budget_bytes\":" << budget << ",\"peak_owned_bytes\":" << gpu.peak
              << ",\"redzones_verified\":true}\n";
    return 0;
}
