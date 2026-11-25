#pragma once
#include "plane.hpp"

struct CylinderGrid {
    Cylinder c{};
    int shells = 1;
};
inline CylinderGrid read_cylinder(const std::string &path, double fov) {
    CylinderGrid grid;
    auto &c = grid.c;
    auto text = small_file(path, 4096);
    if (text.compare(0, 3, "\xef\xbb\xbf") == 0)
        text.erase(0, 3);
    std::istringstream in(text);
    std::string tag, extra;
    auto dim = [&](int &n, int lo) { return plane_number(in, n) && n >= lo && n <= 4096; };
    if (!(in >> tag) || tag != "CFDKCYL1" || !dim(c.w, 3) || !dim(c.h, 1) || !dim(grid.shells, 1))
        throw std::runtime_error("expected CFDKCYL1 columns rows shells");
    auto number = [&](double &x) {
        if (!plane_number(in, x) || !std::isfinite(x) ||
            std::abs(x) > std::numeric_limits<float>::max())
            throw std::runtime_error("expected finite cylinder fields");
    };
    for (auto a : {c.center, c.axis, c.seam})
        for (int i = 0; i < 3; ++i)
            number(a[i]);
    number(c.firstRadius);
    number(c.radialStep);
    number(c.pitch);
    if ((in >> extra) || !(c.firstRadius > 0 && c.pitch > 0 && c.radialStep >= 0) ||
        (grid.shells > 1 && c.radialStep == 0))
        throw std::runtime_error("invalid cylinder radius, pitch or trailing fields");
    auto dot = [](const double *a, const double *b) {
        return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
    };
    double na = std::sqrt(dot(c.axis, c.axis)), nu = std::sqrt(dot(c.seam, c.seam)),
           proj = dot(c.axis, c.seam);
    if (std::abs(na - 1) > 1e-6 || std::abs(nu - 1) > 1e-6 || std::abs(proj) > 1e-6)
        throw std::runtime_error("expected orthonormal axis and seam");
    // Canonicalize accepted decimal rounding before computing the envelope and metadata.
    for (auto &v : c.axis)
        v /= na;
    proj = dot(c.axis, c.seam);
    for (int i = 0; i < 3; ++i)
        c.seam[i] -= proj * c.axis[i];
    nu = std::sqrt(dot(c.seam, c.seam));
    for (auto &v : c.seam)
        v /= nu;
    const double rmax = c.firstRadius + (grid.shells - 1) * c.radialStep;
    for (int i = 0; i < 3; ++i) {
        int j = (i + 1) % 3, k = (i + 2) % 3;
        c.tangent[i] = c.axis[j] * c.seam[k] - c.axis[k] * c.seam[j];
        double extent = std::abs(c.center[i]) + .5 * (c.h - 1) * c.pitch * std::abs(c.axis[i]) +
                        rmax * std::hypot(c.seam[i], c.tangent[i]);
        if (extent > .5 * fov)
            throw std::runtime_error("cylinder outside the field of view");
    }
    return grid;
}
inline void print_cylinder(const CylinderGrid &grid) {
    const auto &c = grid.c;
    std::cout << std::defaultfloat << std::setprecision(17)
              << ",\"format\":\"raw-f32-le-cylinder\",\"shape_rha\":[" << grid.shells << ',' << c.h
              << ',' << c.w << "],\"data_offset_bytes\":0,\"azimuth_step_deg\":" << 360. / c.w
              << ",\"duplicate_endpoint\":false,\"center_mm\":";
    xyz(std::cout, c.center, '[', ']');
    std::cout << ",\"axis_unit\":";
    xyz(std::cout, c.axis, '[', ']');
    std::cout << ",\"seam_unit\":";
    xyz(std::cout, c.seam, '[', ']');
    std::cout << ",\"tangent_unit\":";
    xyz(std::cout, c.tangent, '[', ']');
    std::cout << ",\"first_radius_mm\":" << c.firstRadius << ",\"radial_step_mm\":" << c.radialStep
              << ",\"axial_pitch_mm\":" << c.pitch << std::fixed << std::setprecision(4);
}
