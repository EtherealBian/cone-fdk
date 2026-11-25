#pragma once
#include "cylinder.hpp"

inline bool valid_wall_record(const WallSample &s, double lo, double hi) {
    if (s.reserved || s.status >= WallStatusCount)
        return false;
    if (s.status != Measured)
        return std::isnan(s.inner) && std::isnan(s.outer) && std::isnan(s.thickness);
    if (!std::isfinite(s.outer))
        return false;
    const double tol = 16 * std::numeric_limits<double>::epsilon();
    return s.inner > 0 && s.inner >= lo - tol * lo && s.outer <= hi + tol * hi &&
           s.outer > s.inner && s.thickness > 0 && std::isfinite(s.thickness) &&
           std::abs((s.outer - s.inner) - s.thickness) <=
               16 * std::numeric_limits<double>::epsilon() * s.outer;
}
