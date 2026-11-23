#pragma once
// Windows x64 guarantees SSE2. Inspect exponent bits without doing FP arithmetic:
// both signs of infinity and all NaN payloads must be rejected, even signalling NaNs.
#include <cmath>
#include <cstddef>
#include <emmintrin.h>
inline bool finite_block(const float *p, size_t count) {
    const __m128i exponent = _mm_set1_epi32(0x7f800000);
    __m128i bad = _mm_setzero_si128();
    for (; count >= 4; count -= 4, p += 4) {
        auto x = _mm_castps_si128(_mm_loadu_ps(p));
        bad = _mm_or_si128(bad, _mm_cmpeq_epi32(_mm_and_si128(x, exponent), exponent));
    }
    if (_mm_movemask_epi8(bad))
        return false;
    for (; count; --count, ++p)
        if (!std::isfinite(*p))
            return false;
    return true;
}
