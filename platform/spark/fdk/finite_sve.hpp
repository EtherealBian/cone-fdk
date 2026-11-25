#pragma once
// Bit tests, not floating-point comparisons: preserve all NaN payloads and signed infinities.
#include <arm_sve.h>
#include <asm/hwcap.h>
#include <sys/auxv.h>
#include <cmath>
#include <cstdint>
#include <stdexcept>
inline void require_sve() {
    if (!(getauxval(AT_HWCAP) & HWCAP_SVE)) throw std::runtime_error("Spark SVE required");
}
__attribute__((target("+sve"))) inline unsigned spark_vector_bits() { return svcntb()*8; }
__attribute__((target("+sve"))) inline bool finite_block_sve(const float *p, size_t n) {
    const uint32_t exp = 0x7f800000u;
    const auto all = svptrue_b32();
    svuint32_t maxima = svdup_u32(0);
    // One predicated integer load via reinterpret (no C++ type-punning, no FP arithmetic).
    // Inactive lanes read zero, so a partial vector cannot hide earlier invalid lanes.
    for (size_t i=0; i<n; i+=svcntw()) {
        auto pg=svwhilelt_b32(uint64_t(i),uint64_t(n));
        auto bits=svreinterpret_u32_f32(svld1_f32(pg,p+i));
        maxima=svmax_u32_x(all,maxima,svand_n_u32_x(all,bits,exp));
    }
    return svmaxv_u32(all,maxima)!=exp;
}
inline bool finite_block(const float *p,size_t n) {require_sve(); return finite_block_sve(p,n);}
