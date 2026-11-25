#pragma once
#include "files.hpp"
#include "geometry.hpp"
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>

// Plane is also the CUDA kernel ABI; only this host descriptor knows about layers.
struct PlaneGrid {
    Plane p{};
    int depth = 1;
    double layer[3]{};
    Plane at(int k) const {
        Plane result = p;
        for (int i = 0; i < 3; ++i)
            result.center[i] += (k - .5 * (depth - 1)) * layer[i];
        return result;
    }
};
// Consume whole tokens: stream extraction alone accepts the integer prefix of 2.5.
template <class T> inline bool plane_number(std::istream &input, T &value) {
    std::string token;
    if (!(input >> token))
        return false;
    std::istringstream number(token);
    return (number >> value) && number.eof();
}
inline PlaneGrid read_grid(const std::string &path, float fov, bool stacked) {
    auto text = small_file(path, 4096);
    if (text.compare(0, 3, "\xef\xbb\xbf") == 0)
        text.erase(0, 3);
    std::istringstream input(text);
    std::string tag, extra;
    PlaneGrid grid;
    Plane &p = grid.p;
    auto dimension = [&](int &n) { return plane_number(input, n) && n >= 1 && n <= 4096; };
    if (!(input >> tag) || tag != (stacked ? "CFDKSTACK1" : "CFDKPLANE1") || !dimension(p.w) ||
        !dimension(p.h) || (stacked && !dimension(grid.depth)))
        throw std::runtime_error(stacked ? "expected CFDKSTACK1 width height depth (1..4096)"
                                         : "expected CFDKPLANE1 width height (1..4096)");
    double *vectors[] = {p.center, p.col, p.row, grid.layer};
    const int axes = stacked ? 3 : 2;
    for (int a = 0; a <= axes; ++a)
        for (int i = 0; i < 3; ++i) {
            auto &v = vectors[a][i];
            if (!plane_number(input, v) || !std::isfinite(v) ||
                std::abs(v) > std::numeric_limits<float>::max())
                throw std::runtime_error("expected finite XYZ vectors in mm");
        }
    if (input >> extra)
        throw std::runtime_error("unexpected trailing grid fields");
    auto dot = [](const double *a, const double *b) {
        return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
    };
    double lengths[4]{};
    for (int a = 1; a <= axes; ++a) {
        lengths[a] = std::sqrt(dot(vectors[a], vectors[a]));
        if (lengths[a] < std::numeric_limits<float>::min())
            throw std::runtime_error("grid steps must be nonzero");
        for (int b = 1; b < a; ++b)
            if (std::abs(dot(vectors[a], vectors[b])) > 1e-5 * lengths[a] * lengths[b])
                throw std::runtime_error("grid steps must be orthogonal");
    }
    for (int i = 0; i < 3; ++i) {
        // Largest absolute coordinate of any sample center, including singleton axes.
        double extent = std::abs(p.center[i]) + .5 * (p.w - 1) * std::abs(p.col[i]) +
                        .5 * (p.h - 1) * std::abs(p.row[i]) +
                        .5 * (grid.depth - 1) * std::abs(grid.layer[i]);
        if (extent > .5 * fov)
            throw std::runtime_error("sample centers outside the field of view");
    }
    return grid;
}
inline void xyz(std::ostream &out, const double *v, char l, char r) {
    out << l << v[0] << ',' << v[1] << ',' << v[2] << r;
}
inline std::string nrrd_header(const PlaneGrid &g) {
    std::ostringstream out;
    const auto &p = g.p;
    auto first = g.at(0);
    for (int i = 0; i < 3; ++i)
        first.center[i] -= .5 * (p.w - 1) * p.col[i] + .5 * (p.h - 1) * p.row[i];
    out << std::setprecision(17)
        << "NRRD0005\ntype: float\ndimension: 3\nspace: 3D-right-handed\nsizes: " << p.w << ' '
        << p.h << ' ' << g.depth << "\nspace directions: ";
    xyz(out, p.col, '(', ')');
    out << ' ';
    xyz(out, p.row, '(', ')');
    out << ' ';
    xyz(out, g.layer, '(', ')');
    out << "\nspace units: \"mm\" \"mm\" \"mm\"\nspace origin: ";
    xyz(out, first.center, '(', ')');
    out << "\nkinds: domain domain domain\nendian: little\nencoding: raw\n\n";
    return out.str();
}
inline void print_grid(const PlaneGrid &g, bool stacked, bool nrrd, size_t header) {
    const Plane &p = g.p;
    if (stacked) {
        std::cout << ",\"format\":\"" << (nrrd ? "nrrd-f32-le-stack" : "raw-f32-le-stack")
                  << "\",\"data_offset_bytes\":" << header << ",\"shape_zyx\":[" << g.depth << ',';
    } else {
        std::cout << ",\"format\":\"raw-f32-le-plane\",\"shape_yx\":[";
    }
    std::cout << p.h << ',' << p.w << "],\"center_mm\":" << std::defaultfloat
              << std::setprecision(17);
    xyz(std::cout, p.center, '[', ']');
    std::cout << ",\"col_step_mm\":";
    xyz(std::cout, p.col, '[', ']');
    std::cout << ",\"row_step_mm\":";
    xyz(std::cout, p.row, '[', ']');
    if (stacked) {
        std::cout << ",\"slice_step_mm\":";
        xyz(std::cout, g.layer, '[', ']');
    }
    std::cout << std::fixed << std::setprecision(4);
}
