// Optional native companion. Keep the released executable/PTX and its wall ABI unchanged.
#include "gpu.hpp"
#include "wall_record.hpp"
#include <array>
#include <map>

namespace {
constexpr const char *usage = "wall-pack --input atlas.f32 --cylinder grid.txt --ptx kernels.ptx "
                              "--output levels.wallpack --budget MiB --level X [--level Y ...]";

template <class T> T number(const std::string &s) {
    std::istringstream in(s);
    T result;
    std::string extra;
    if (!plane_number(in, result) || (in >> extra))
        throw std::runtime_error("invalid number: " + s);
    return result;
}

void run(const std::vector<std::string> &args) {
    std::map<std::string, std::string> opt;
    std::vector<double> levels;
    for (size_t i = 1; i < args.size(); i += 2) {
        if (i + 1 == args.size())
            throw std::runtime_error(usage);
        const auto &k = args[i];
        if (k == "--level") {
            const auto v = number<double>(args[i + 1]);
            if (!(v > 0 && std::isfinite(v)) || (!levels.empty() && v <= levels.back()) ||
                levels.size() == 16)
                throw std::runtime_error("expected 1..16 increasing positive finite levels");
            levels.push_back(v);
        } else if ((k == "--input" || k == "--cylinder" || k == "--ptx" || k == "--output" ||
                    k == "--budget") &&
                   !opt.count(k)) {
            opt[k] = args[i + 1];
        } else
            throw std::runtime_error(usage);
    }
    if (opt.size() != 5 || levels.empty())
        throw std::runtime_error(usage);
    const int mib = number<int>(opt.at("--budget"));
    if (mib < 1 || mib > 65536)
        throw std::runtime_error("invalid budget");
    const size_t budget = size_t(mib) << 20;
    auto dest = absolute(opt.at("--output"));
    for (auto k : {"--input", "--cylinder", "--ptx"})
        if (same_file(dest, absolute(opt.at(k))))
            throw std::runtime_error("output aliases an input");
    const auto grid = read_cylinder(opt.at("--cylinder"), std::numeric_limits<double>::infinity());
    const auto &c = grid.c;
    const int d = grid.shells, n = c.w * c.h;
    const double hi = c.firstRadius + (d - 1) * c.radialStep;
    if (d < 3 || !std::isfinite(hi))
        throw std::runtime_error("invalid radial range");
    for (int s = 1; s < d; ++s)
        if (!(c.firstRadius + s * c.radialStep > c.firstRadius + (s - 1) * c.radialStep))
            throw std::runtime_error("indistinct radial knots");
    File input(absolute(opt.at("--input")));
    const size_t inputBytes = size_t(n) * d * sizeof(float),
                 outputBytes = size_t(n) * levels.size() * sizeof(WallSample);
    if (input.size() != inputBytes)
        throw std::runtime_error("atlas size mismatch");
    Gpu gpu(small_file(opt.at("--ptx"), 16u << 20).c_str());
    auto kernel = gpu.kernel("radial_wall");
    size_t free, total;
    CK(gpu.cuMemGetInfo_v2(&free, &total));
    const size_t reserve = 128u << 20, guards = 4 * Buf::pad;
    const size_t cap = std::min<size_t>({budget, 16u << 20, free > reserve ? free - reserve : 0});
    const size_t stride = d * sizeof(float) + sizeof(WallSample);
    if (cap < guards + stride)
        throw std::runtime_error("budget cannot fit one profile");
    const int tile = int(std::min<size_t>(n, (cap - guards) / stride));
    std::vector<float> host(size_t(tile) * d);
    std::vector<WallSample> result(tile);
    Buf atlas(gpu, host.size() * sizeof(float)), wall(gpu, result.size() * sizeof(WallSample));
    std::vector<std::array<size_t, WallStatusCount>> counts(levels.size(),
                                                            std::array<size_t, WallStatusCount>{});
    size_t read = 0, uploaded = 0, downloaded = 0, written = 0, launches = 0;
    Output output(dest);
    for (int pos = 0; pos < n; pos += tile) {
        const int rays = std::min(tile, n - pos);
        for (int k = 0; k < d; ++k) {
            input.seek((size_t(k) * n + pos) * sizeof(float));
            size_t bytes = rays * sizeof(float);
            input.transfer(host.data() + size_t(k) * rays, bytes, false);
            read += bytes;
        }
        const size_t inBytes = size_t(d) * rays * sizeof(float),
                     outBytes = rays * sizeof(WallSample);
        CK(gpu.cuMemcpyHtoD_v2(atlas.p, host.data(), inBytes));
        uploaded += inBytes;
        for (size_t i = 0; i < levels.size(); ++i) {
            gpu.launch(kernel, (rays + 127) / 128, 1, 1, 128, 1, atlas.p, wall.p, rays, d,
                       c.firstRadius, c.radialStep, levels[i]);
            ++launches;
            CK(gpu.cuMemcpyDtoH_v2(result.data(), wall.p, outBytes));
            downloaded += outBytes;
            for (int j = 0; j < rays; ++j) {
                if (!valid_wall_record(result[j], c.firstRadius, hi))
                    throw std::runtime_error("invalid wall record from GPU");
                ++counts[i][result[j].status];
            }
            output.file->seek((i * n + pos) * sizeof(WallSample));
            output.file->transfer(result.data(), outBytes, true);
            written += outBytes;
        }
    }
    atlas.verify();
    wall.verify();
    if (read != inputBytes || uploaded != inputBytes || downloaded != outputBytes ||
        written != outputBytes || output.file->size() != outputBytes || gpu.peak > budget)
        throw std::runtime_error("payload accounting mismatch");
    output.publish();
    std::cout << std::setprecision(17) << "{\"format\":\"CFDK-wall-pack-v1\",\"gpu\":\"" << gpu.name
              << "\",\"shape_ha\":[" << c.h << ',' << c.w << "],\"shells\":" << d
              << ",\"levels_per_mm\":[";
    for (size_t i = 0; i < levels.size(); ++i)
        std::cout << (i ? "," : "") << levels[i];
    std::cout << "],\"status_counts_by_level\":[";
    for (size_t i = 0; i < levels.size(); ++i) {
        std::cout << (i ? ",[" : "[");
        for (unsigned s = 0; s < WallStatusCount; ++s)
            std::cout << (s ? "," : "") << counts[i][s];
        std::cout << ']';
    }
    std::cout << "],\"atlas_bytes_read\":" << read << ",\"atlas_payload_h2d_bytes\":" << uploaded
              << ",\"wall_payload_d2h_bytes\":" << downloaded << ",\"output_bytes\":" << written
              << ",\"tiles\":" << (n + tile - 1) / tile << ",\"kernel_launches\":" << launches
              << ",\"budget_bytes\":" << budget << ",\"peak_owned_bytes\":" << gpu.peak
              << ",\"redzones_verified\":true}\n";
}
} // namespace

int main() {
    try {
        auto args = arguments();
        if (args.size() == 2 && args[1] == "--help") {
            std::cout << usage << '\n';
            return 0;
        }
        run(args);
        return 0;
    } catch (const std::exception &e) {
        std::cerr << "wall-pack: " << e.what() << '\n';
        return 2;
    }
}
