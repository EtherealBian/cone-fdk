#include "../src/files.hpp"
#include "../src/finite.hpp"
#include "driver.hpp"
#include <array>
#include <climits>
#include <chrono>
#include <iomanip>
#include <map>
#include <set>
#include <sstream>
using U = unsigned long long;
using Args = std::map<std::string, std::string>;
static void require(bool b, const char *msg) {
    if (!b)
        throw std::runtime_error(msg);
}
static int integer(const Args &a, const char *k, int hi) {
    const auto &s = a.at(k);
    require(!s.empty() && s.find_first_not_of("0123456789") == s.npos, "expected positive integer");
    auto v = std::stoull(s);
    require(v > 0 && v <= U(hi), "integer outside supported range");
    return int(v);
}
static double number(const Args &a, const char *k) {
    const auto &s = a.at(k);
    size_t n = 0;
    auto v = std::stod(s, &n);
    require(n == s.size() && std::isfinite(v), "expected a finite number");
    return v;
}
struct Part {
    U seed = 0, n = 0, sx = 0, sy = 0, sz = 0;
    int parent = -1;
    unsigned faces = 0;
    void add(int x, int y, int z, unsigned f) {
        ++n;
        sx += x;
        sy += y;
        sz += z;
        faces |= f;
    }
    void absorb(const Part &b) {
        n += b.n;
        sx += b.sx;
        sy += b.sy;
        sz += b.sz;
        faces |= b.faces;
    }
};
struct Components {
    std::vector<Part> v;
    int make(U seed) {
        require(v.size() < INT_MAX, "too many slab components for host index");
        Part p;
        p.seed = seed;
        p.parent = int(v.size());
        v.push_back(p);
        return p.parent;
    }
    int root(int i) {
        while (i != v[i].parent) {
            v[i].parent = v[v[i].parent].parent;
            i = v[i].parent;
        }
        return i;
    }
    void join(int a, int b) {
        a = root(a);
        b = root(b);
        if (a == b)
            return;
        if (a > b)
            std::swap(a, b);
        v[b].parent = a;
        v[a].absorb(v[b]);
    }
};
static void stitch(Components &c, const std::vector<int> &prev, const std::vector<int> &curr,
                   int nx, int ny, int conn) {
    int r = (conn == 26);
    for (int y = 0; y < ny; ++y)
        for (int x = 0; x < nx; ++x) {
            int a = curr[y * nx + x];
            if (a < 0)
                continue;
            for (int j = std::max(0, y - r); j <= std::min(ny - 1, y + r); ++j)
                for (int i = std::max(0, x - r); i <= std::min(nx - 1, x + r); ++i) {
                    int b = prev[j * nx + i];
                    if (b >= 0)
                        c.join(a, b);
                }
        }
}
static void writeText(Output &out, std::string s) {
    out.file->transfer(&s[0], s.size(), true);
}
static int analyze() {
    auto av = arguments();
    if (av.size() == 2 && av[1] == "--help") {
        std::cout
            << "cone-pores --input volume.f32 --output pores.csv --nx X --ny Y --nz Z --sx um --sy "
               "um --sz um --threshold T [--connectivity 6|26] [--filter none|median-xy] "
               "[--budget-mib M] [--slab Z] [--ptx pores.ptx]\nInput: little-endian float32, X "
               "fastest. Analyze a rectangular in-material ROI. Dark phase <= T, including clipped "
               "components. CSV coordinates are crop-local. JSON summary on stdout.\n";
        return 0;
    }
    Args a{{"--connectivity", "6"},
           {"--budget-mib", "64"},
           {"--slab", "1000000"},
           {"--filter", "none"}};
    std::set<std::string> seen;
    std::set<std::string> valid{
        "--input", "--output",    "--nx",           "--ny",     "--nz",         "--sx",   "--sy",
        "--sz",    "--threshold", "--connectivity", "--filter", "--budget-mib", "--slab", "--ptx"};
    for (size_t i = 1; i < av.size(); i += 2) {
        require(i + 1 < av.size() && valid.count(av[i]) && seen.insert(av[i]).second,
                "unknown, duplicated or incomplete option");
        a[av[i]] = av[i + 1];
    }
    for (auto k :
         {"--input", "--output", "--nx", "--ny", "--nz", "--sx", "--sy", "--sz", "--threshold"})
        require(a.count(k), "missing required option; see --help");
    int nx = integer(a, "--nx", 1000000), ny = integer(a, "--ny", 1000000),
        nz = integer(a, "--nz", 1000000);
    int conn = integer(a, "--connectivity", 26);
    require(conn == 6 || conn == 26, "connectivity must be 6 or 26");
    auto filter = a.at("--filter");
    require(filter == "median-xy" || filter == "none", "unknown filter");
    double dx = number(a, "--sx"), dy = number(a, "--sy"), dz = number(a, "--sz");
    for (auto d : {dx, dy, dz})
        require(d >= 1.e-9 && d <= 1.e9, "invalid spacing in micrometres");
    float t = float(number(a, "--threshold"));
    require(std::isfinite(t), "threshold outside float32");
    U plane = U(nx) * ny, total = plane * nz;
    require(plane <= INT_MAX - 255 && total <= 1000000000000ull, "volume outside index limits");
    U budget = U(integer(a, "--budget-mib", 1048576)) << 20;
    int cap = integer(a, "--slab", 1000000);
    auto input = absolute(a.at("--input")), dest = absolute(a.at("--output"));
    wchar_t self[32768];
    DWORD len = GetModuleFileNameW(nullptr, self, 32768);
    require(len && len < 32768, "cannot locate executable");
    std::wstring exe(self, len);
    auto ptx = a.count("--ptx") ? absolute(a.at("--ptx"))
                                : exe.substr(0, exe.find_last_of(L"\\/") + 1) + L"pores.ptx";
    require(!same_file(input, dest) && !same_file(ptx, dest) && !same_file(exe, dest),
            "output aliases input or program");
    File in(input);
    require(in.size() == total * 4, "wrong float32 volume length");
    auto start = std::chrono::steady_clock::now();
    LabelsGpu gpu(small_file(utf8(ptx.c_str()), 1 << 20));
    size_t free = 0, all = 0;
    CK(gpu.cuMemGetInfo_v2(&free, &all));
    U effective = std::min<U>(budget, free - free / 4);
    int slab = int(std::min<U>({U(nz), U(cap), effective / (8 * plane), U(INT_MAX - 255) / plane}));
    require(slab > 0, "not enough device budget for one XY plane");
    size_t maxn = size_t(plane) * slab;
    CK(gpu.cuMemAlloc_v2(&gpu.io, maxn * 4));
    CK(gpu.cuMemAlloc_v2(&gpu.parents, maxn * 4));
    std::vector<float> data(maxn);
    std::vector<int> labels(maxn), prev;
    Components comps;
    for (int z0 = 0; z0 < nz; z0 += slab) {
        int depth = std::min(nz - z0, slab), n = int(plane * depth);
        in.transfer(data.data(), size_t(n) * 4, false);
        require(finite_block(data.data(), n), "non-finite input voxel");
        gpu.label(data.data(), labels.data(), nx, ny, n, t, filter == "median-xy", conn);
        // Minimum-index CUDA roots occur before all their members. Reuse their label slots as
        // global handles.
        for (int i = 0; i < n; ++i) {
            int local = labels[i];
            if (local < 0)
                continue;
            require(local <= i, "invalid CUDA label");
            int g = local == i ? comps.make(U(z0) * plane + i) : labels[local];
            labels[i] = g;
            int x = i % nx, y = i / nx % ny, z = z0 + i / int(plane);
            unsigned faces = (x == 0) | ((x == nx - 1) << 1) | ((y == 0) << 2) |
                             ((y == ny - 1) << 3) | ((z == 0) << 4) | ((z == nz - 1) << 5);
            comps.v[g].add(x, y, z, faces);
        }
        if (z0)
            stitch(comps, prev, labels, nx, ny, conn);
        prev.assign(labels.begin() + n - plane, labels.begin() + n);
    }
    Output out(dest);
    writeText(out, "id,voxels,volume_um3,equivalent_diameter_um,centroid_x_um,centroid_y_um,"
                   "centroid_z_um,seed_x,seed_y,seed_z,faces\n");
    U pores = 0, vox = 0, closed = 0, closedvox = 0;
    std::array<U, 3> through{};
    std::ostringstream rows;
    rows << std::setprecision(17);
    for (size_t i = 0; i < comps.v.size(); ++i) {
        auto &p = comps.v[i];
        if (p.parent != int(i))
            continue;
        ++pores;
        vox += p.n;
        if (!p.faces) {
            ++closed;
            closedvox += p.n;
        }
        for (int k = 0; k < 3; ++k)
            if (((p.faces >> (2 * k)) & 3) == 3)
                ++through[k];
        auto volume = p.n * dx * dy * dz;
        rows << p.seed + 1 << ',' << p.n << ',' << volume << ','
             << std::cbrt(6 * volume / 3.14159265358979323846) << ','
             << (double(p.sx) / p.n + .5) * dx << ',' << (double(p.sy) / p.n + .5) * dy << ','
             << (double(p.sz) / p.n + .5) * dz << ',' << p.seed % nx << ',' << p.seed / nx % ny
             << ',' << p.seed / plane << ',' << p.faces << '\n';
        if ((pores & 1023) == 0) {
            writeText(out, rows.str());
            rows.str("");
        }
    }
    if (rows.tellp() > 0)
        writeText(out, rows.str());
    out.publish();
    std::cout << std::setprecision(17) << "{\"gpu\":\"" << gpu.name << "\",\"shape_zyx\":[" << nz
              << ',' << ny << ',' << nx << "],\"spacing_xyz_um\":[" << dx << ',' << dy << ',' << dz
              << "],\"threshold_f32\":" << t << ",\"filter\":\"" << filter
              << "\",\"connectivity\":" << conn << ",\"slab_depth\":" << slab
              << ",\"device_buffer_bytes\":" << 8 * maxn << ",\"roi_voxels\":" << total
              << ",\"pore_voxels\":" << vox << ",\"porosity\":" << double(vox) / total
              << ",\"components\":" << pores << ",\"interior_components\":" << closed
              << ",\"interior_pore_voxels\":" << closedvox << ",\"spanning_xyz\":[" << through[0]
              << ',' << through[1] << ',' << through[2] << "],\"elapsed_s\":"
              << std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count()
              << "}\n";
    return 0;
}
int main() {
    try {
        return analyze();
    } catch (const std::exception &e) {
        std::cerr << "cone-pores: " << e.what() << '\n';
        return 1;
    }
}
