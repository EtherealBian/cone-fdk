// One read of every ternary byte, retaining only a band, the prior row and a lookahead row.
#include "files.hpp"
#include "gpu.hpp"
#include <map>

namespace {
constexpr auto usage =
    "periodic-interfaces --input masks.u8 --output faces.u8 --ptx interfaces.ptx "
    "--height H --width W --maps K --band ROWS --budget MiB";
int number(const std::string &s, int lo, int hi) {
    if (s.empty() || s.size() > 8 || s.find_first_not_of("0123456789") != std::string::npos)
        throw std::runtime_error("invalid integer");
    const int n = std::stoi(s);
    if (n < lo || n > hi)
        throw std::runtime_error("argument outside supported range");
    return n;
}
size_t aligned(size_t n) {
    return (n + 3) & ~size_t(3);
}
void run(const std::vector<std::string> &args) {
    std::map<std::string, std::string> opt;
    for (size_t i = 1; i < args.size(); i += 2)
        if (i + 1 >= args.size() || !opt.emplace(args[i], args[i + 1]).second)
            throw std::runtime_error(usage);
    for (auto key :
         {"--input", "--output", "--ptx", "--width", "--height", "--maps", "--band", "--budget"})
        if (!opt.count(key))
            throw std::runtime_error(usage);
    if (opt.size() != 8)
        throw std::runtime_error(usage);
    const int w = number(opt["--width"], 3, 4096), h = number(opt["--height"], 1, 4096),
              maps = number(opt["--maps"], 1, 4096), band = number(opt["--band"], 1, 4096),
              cap = std::min(h, band);
    const size_t budget = size_t(number(opt["--budget"], 1, 65536)) << 20,
                 total = size_t(maps) * h * w, maskcap = size_t(cap + 2) * w,
                 outcap = size_t(cap) * w,
                 owned = aligned(maskcap) + aligned(outcap) + 4 * Buf::pad;
    static_assert(sizeof(size_t) == 8, "requires Windows x64");
    if (owned > budget)
        throw std::runtime_error("device budget cannot fit guarded band");
    const auto dst = absolute(opt["--output"]);
    if (GetFileAttributesW(dst.c_str()) != INVALID_FILE_ATTRIBUTES)
        throw std::runtime_error("destination already exists");
    File input(absolute(opt["--input"]));
    if (input.size() != total)
        throw std::runtime_error("input size mismatch");
    Gpu gpu(small_file(opt["--ptx"], 16u << 20).c_str());
    auto kernel = gpu.kernel("encode_interfaces");
    Buf mask(gpu, aligned(maskcap)), faces(gpu, aligned(outcap));
    mask.verify();
    faces.verify();
    std::vector<unsigned char> plane(maskcap), code(outcap), previous(w), next(w);
    Output output(dst);
    size_t reads = 0, transfers = 0, launches = 0;
    auto take = [&](unsigned char *p, size_t n) {
        if (!n)
            return;
        input.transfer(p, n, false);
        reads += n;
        if (*std::max_element(p, p + n) > 2)
            throw std::runtime_error("invalid ternary byte");
    };
    for (int m = 0; m < maps; ++m) {
        std::fill(previous.begin(), previous.end(), 3);
        for (int y = 0; y < h; y += band) {
            const int rows = std::min(band, h - y), cells = rows * w, span = (rows + 2) * w;
            std::copy(previous.begin(), previous.end(), plane.begin());
            if (y == 0)
                take(plane.data() + w, cells);
            else {
                std::copy(next.begin(), next.end(), plane.begin() + w);
                take(plane.data() + 2 * w, cells - w);
            }
            if (y + rows < h)
                take(next.data(), w);
            else
                std::fill(next.begin(), next.end(), 3);
            std::copy(next.begin(), next.end(), plane.begin() + w + cells);
            std::copy_n(plane.begin() + cells, w, previous.begin());
            CK(gpu.cuMemcpyHtoD_v2(mask.p, plane.data(), span));
            transfers += span;
            gpu.launch(kernel, (cells + 127) / 128, 1, 1, 128, 1, mask.p, faces.p, w, cells);
            ++launches;
            CK(gpu.cuMemcpyDtoH_v2(code.data(), faces.p, cells));
            output.file->transfer(code.data(), cells, true);
        }
    }
    mask.verify();
    faces.verify();
    if (reads != total || output.file->size() != total || gpu.peak != owned ||
        transfers != total + 2 * size_t(w) * launches)
        throw std::runtime_error("stream or allocation accounting mismatch");
    output.file->finish();
    if (!MoveFileExW(output.tmp.c_str(), dst.c_str(), MOVEFILE_WRITE_THROUGH))
        throw std::runtime_error("cannot publish fresh output");
    output.published = true;
    std::cout << "{\"format\":\"CFDK-periodic-interfaces-v1\",\"gpu\":\"" << gpu.name
              << "\",\"driver\":" << gpu.driver << ",\"shape_ha\":[" << h << "," << w
              << "],\"maps\":" << maps << ",\"band_rows\":" << band << ",\"read_bytes\":" << reads
              << ",\"h2d_bytes\":" << transfers << ",\"d2h_bytes\":" << total
              << ",\"kernel_launches\":" << launches << ",\"peak_owned_bytes\":" << gpu.peak
              << ",\"budget_bytes\":" << budget << ",\"redzones_verified\":true}\n";
}
} // namespace
int main() {
    try {
        auto a = arguments();
        if (a.size() == 2 && a[1] == "--help") {
            std::cout << usage << '\n';
            return 0;
        }
        run(a);
        return 0;
    } catch (const std::exception &e) {
        std::cerr << "periodic-interfaces: " << e.what() << '\n';
        return 2;
    }
}
