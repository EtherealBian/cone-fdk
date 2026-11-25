// Standalone preclassified mask -> exact periodic labels. Does not classify wall records.
#include "files.hpp"
#include "gpu.hpp"
#include <map>

namespace {
constexpr auto usage = "periodic-labels --input masks.u8 --output labels.i32 --ptx labels.ptx "
                       "--height H --width W --maps K --budget MiB";
int bounded(const std::string &s, int lo, int hi) {
    if (s.empty() || s.size() > 9 || s.find_first_not_of("0123456789") != std::string::npos)
        throw std::runtime_error("invalid unsigned integer");
    int v = std::stoi(s);
    if (v < lo || v > hi)
        throw std::runtime_error("integer outside supported range");
    return v;
}
void run(const std::vector<std::string> &args) {
    std::map<std::string, std::string> opt;
    for (size_t i = 1; i < args.size(); i += 2) {
        if (i + 1 == args.size() || !opt.emplace(args[i], args[i + 1]).second)
            throw std::runtime_error(usage);
    }
    for (auto k : {"--input", "--output", "--ptx", "--height", "--width", "--maps", "--budget"})
        if (!opt.count(k))
            throw std::runtime_error(usage);
    if (opt.size() != 7)
        throw std::runtime_error(usage);
    const int h = bounded(opt["--height"], 1, 4096), w = bounded(opt["--width"], 3, 4096),
              k = bounded(opt["--maps"], 1, 4096), n = h * w;
    const size_t budget = size_t(bounded(opt["--budget"], 1, 65536)) << 20;
    const size_t padded = (size_t(n) + 3) & ~size_t(3), ints = size_t(n) * sizeof(int),
                 owned = padded + 2 * ints + 6 * Buf::pad;
    static_assert(sizeof(int) == 4 && sizeof(size_t) == 8, "requires Windows x64");
    if (budget < owned)
        throw std::runtime_error("device budget cannot fit one guarded plane");
    const auto dst = absolute(opt["--output"]);
    if (GetFileAttributesW(dst.c_str()) != INVALID_FILE_ATTRIBUTES)
        throw std::runtime_error("destination already exists");
    File in(absolute(opt["--input"]));
    if (in.size() != size_t(k) * n)
        throw std::runtime_error("input size mismatch");
    Gpu gpu(small_file(opt["--ptx"], 16u << 20).c_str());
    auto init = gpu.kernel("init_forest"), link = gpu.kernel("link_cylinder"),
         emit = gpu.kernel("emit_canonical");
    Buf mask(gpu, padded), parent(gpu, ints), label(gpu, ints);
    std::vector<unsigned char> data(n);
    std::vector<int> result(n);
    Output out(dst);
    auto verify = [&]() {
        mask.verify();
        parent.verify();
        label.verify();
    };
    verify();
    Timer events(gpu);
    double kernel_ms = 0;
    for (int m = 0; m < k; ++m) {
        in.transfer(data.data(), n, false);
        if (*std::max_element(data.begin(), data.end()) > 2)
            throw std::runtime_error("invalid ternary byte in scenario " + std::to_string(m));
        CK(gpu.cuMemcpyHtoD_v2(mask.p, data.data(), n));
        int blocks = (n + 127) / 128;
        events.start();
        gpu.launch(init, blocks, 1, 1, 128, 1, parent.p, n);
        gpu.launch(link, blocks, 1, 1, 128, 1, mask.p, parent.p, n, w);
        gpu.launch(emit, blocks, 1, 1, 128, 1, mask.p, parent.p, label.p, n);
        kernel_ms += events.stop();
        CK(gpu.cuMemcpyDtoH_v2(result.data(), label.p, ints));
        out.file->transfer(result.data(), ints, true);
    }
    verify();
    if (gpu.peak != owned || out.file->size() != size_t(k) * ints)
        throw std::runtime_error("payload accounting mismatch");
    // Unlike the legacy Output::publish(), do not replace a concurrent publisher.
    out.file->finish();
    if (!MoveFileExW(out.tmp.c_str(), dst.c_str(), MOVEFILE_WRITE_THROUGH))
        throw std::runtime_error("cannot publish fresh output");
    out.published = true;
    std::cout << "{\"format\":\"CFDK-periodic-cuda-labels-v1\",\"gpu\":\"" << gpu.name
              << "\",\"driver\":" << gpu.driver << ",\"shape_ha\":[" << h << ',' << w
              << "],\"maps\":" << k << ",\"h2d_payload_bytes\":" << size_t(k) * n
              << ",\"d2h_payload_bytes\":" << size_t(k) * ints << ",\"kernel_launches\":" << 3 * k
              << ",\"gpu_label_event_ms\":" << kernel_ms << ",\"peak_owned_bytes\":" << gpu.peak
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
        std::cerr << "periodic-labels: " << e.what() << '\n';
        return 2;
    }
}
