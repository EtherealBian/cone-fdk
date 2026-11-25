#pragma once
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <cmath>
inline bool finite_block(const float *p, size_t n){for(size_t i=0;i<n;i++){uint32_t u;std::memcpy(&u,p+i,4);if((u&0x7f800000u)==0x7f800000u)return false;}return true;}
