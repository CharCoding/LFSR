#include <bit>
#include <cassert>
#include <cstdint>
#include <cstdio>

#include "acprimpolyopt.hpp"
enum POLY_TYPE { REDUCIBLE,
                 IRREDUCIBLE,
                 PRIMITIVE };
constexpr __uint128_t products[9] = {  // (2^128 - 1) / p where p is a prime factor of 2^128 - 1
    (__uint128_t)0x5555555555555555ULL << 64 | 0x5555555555555555ULL,
    (__uint128_t)0x3333333333333333ULL << 64 | 0x3333333333333333ULL,
    (__uint128_t)0x0f0f0f0f0f0f0f0fULL << 64 | 0x0f0f0f0f0f0f0f0fULL,
    (__uint128_t)0x00ff00ff00ff00ffULL << 64 | 0x00ff00ff00ff00ffULL,
    (__uint128_t)0x0000ffff0000ffffULL << 64 | 0x0000ffff0000ffffULL,
    (__uint128_t)0x00663d80ff99c27fULL << 64 | 0x00663d80ff99c27fULL,
    (__uint128_t)0x00000280fffffd7fULL << 64 | 0x00000280fffffd7fULL,
    (__uint128_t)0x00003d30f19cd100ULL << 64 | 0xffffc2cf0e632effULL,
    (__uint128_t)0x0000000000042f00ULL << 64 | 0xfffffffffffbd0ffULL};
POLY_TYPE isPrimitive(const uint64_t poly) {
  __uint128_t cache[127] = {
      (__uint128_t)1ULL << 126,  // 2^1 - 1
      (__uint128_t)1ULL << 124,  // 2^2 - 1
      (__uint128_t)1ULL << 120,  // 2^3 - 1
      (__uint128_t)1ULL << 112,  // 2^4 - 1
      (__uint128_t)1ULL << 96,   // 2^5 - 1
      (__uint128_t)1ULL << 64,   // 2^6 - 1
      1ULL,                      // 2^7 - 1
  };
  multiply m(poly);
  __m128i xm1 = _mm_set_epi64x(0, 1);  // x^127
  int i = 7;
  do {
    xm1 = m.sqp1(xm1);
    cache[i] = (__uint128_t)_mm_extract_epi64(xm1, 1) << 64 | _mm_extract_epi64(xm1, 0);
  } while (++i < 127);
  // printm128(xm1);
  if (_mm_test_all_zeros(_mm_set_epi64x(0x7fffffffffffffffULL, 0xffffffffffffffffULL), xm1)) {
    i = 8;
    do {
      __uint128_t p = products[i];

    } while (--i);
    return POLY_TYPE::PRIMITIVE;
  }
  return POLY_TYPE::REDUCIBLE;
}

constexpr int popcount(__uint128_t x) {
  uint64_t hi = x >> 64;
  if (hi) {
    return std::popcount(hi) + 64;
  }
  return std::popcount(uint64_t(x));
}
int main() {
  constexpr uint64_t be1le1 = 0x8b5b159d0276d09dULL,
                     be1le0 = 0xf5fa1820020c2fd7ULL,
                     be0le1 = be1le0 << 1 | be1le0 >> 63,
                     be0le0 = be1le0 >> 1 | be1le0 << 63;
  // const char* endianness = "not primitive in LE\n";
  // printf("%016lx is %s", be1le1, endianness + isPrimitive(be1le1) * 4);
  // printf("%016lx is %s", be1le0, endianness + isPrimitive(be1le0) * 4);
  // printf("%016lx is %s", be0le1, endianness + isPrimitive(be0le1) * 4);
  // printf("%016lx is %s", be0le0, endianness + isPrimitive(be0le0) * 4);
  __uint128_t p = products[8];
  while (p > 1) {
    p -= (__uint128_t)1 << popcount(p);
  }
}