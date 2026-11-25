#pragma once
#include "files.hpp"
#include "geometry.hpp"
#include <cmath>
#include <iomanip>
#include <limits>
#include <sstream>

// Consume an entire whitespace-delimited token: stream extraction alone accepts
// the integer prefix of a fractional dimension as the dimension itself.
template <class T> inline bool plane_number(std::istream &input, T &value) {
    std::string token;
    if (!(input >> token))
        return false;
    std::istringstream number(token);
    return (number >> value) && number.eof();
}
inline Plane read_plane(const std::string &path, float fov) {
    auto text = small_file(path, 4096);
    if (text.compare(0, 3, "\xef\xbb\xbf") == 0)
        text.erase(0, 3);
    std::istringstream input(text);
    std::string tag, extra;
    Plane p{};
    if (!(input >> tag) || !plane_number(input, p.w) || !plane_number(input, p.h) ||
        tag != "CFDKPLANE1" || p.w < 1 || p.w > 4096 || p.h < 1 || p.h > 4096)
        throw std::runtime_error("expected CFDKPLANE1 width height (1..4096)");
    for (double *v : {p.center, p.col, p.row})
        for (int i = 0; i < 3; ++i)
            if (!plane_number(input, v[i]) || !std::isfinite(v[i]) ||
                std::abs(v[i]) > std::numeric_limits<float>::max())
                throw std::runtime_error("expected finite XYZ vectors in mm");
    if (input >> extra)
        throw std::runtime_error("unexpected trailing plane fields");
    double cc = 0, rr = 0, cr = 0;
    for (int i = 0; i < 3; ++i) {
        cc += p.col[i] * p.col[i];
        rr += p.row[i] * p.row[i];
        cr += p.col[i] * p.row[i];
        // Maximum absolute coordinate over all pixel centers, including singleton axes.
        double extent = std::abs(p.center[i]) + .5 * (p.w - 1) * std::abs(p.col[i]) +
                        .5 * (p.h - 1) * std::abs(p.row[i]);
        if (extent > .5 * fov)
            throw std::runtime_error("plane pixel centers outside the field of view");
    }
    double col = std::sqrt(cc), row = std::sqrt(rr);
    if (col < std::numeric_limits<float>::min() || row < std::numeric_limits<float>::min() ||
        std::abs(cr) > 1e-5 * col * row)
        throw std::runtime_error("plane pixel steps must be nonzero and orthogonal");
    return p;
}
inline void print_plane(const Plane &p) {
    auto xyz = [](const double *v) {
        std::cout << '[' << v[0] << ',' << v[1] << ',' << v[2] << ']';
    };
    std::cout << ",\"format\":\"raw-f32-le-plane\",\"shape_yx\":[" << p.h << ',' << p.w
              << "],\"center_mm\":" << std::defaultfloat << std::setprecision(17);
    xyz(p.center);
    std::cout << ",\"col_step_mm\":";
    xyz(p.col);
    std::cout << ",\"row_step_mm\":";
    xyz(p.row);
    std::cout << std::fixed << std::setprecision(4);
}
