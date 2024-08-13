#include <x86intrin.h>

#include <cstdint>
#include <cstdio>
// Little endian
constexpr __uint128_t xn_1 = 1;                       // LSB represents x^127
constexpr __uint128_t x0 = (__uint128_t)1ULL << 127;  // MSB represents x^0
constexpr __uint128_t x1 = (__uint128_t)1ULL << 126;  // shifts to the right
void print128(__uint128_t x) {
  printf("%016lx %016lx\n", (uint64_t)(x >> 64), (uint64_t)x);
}
void printm128(__m128i x) {
  printf("%016lx %016lx ", (uint64_t)_mm_extract_epi64(x, 1),
         (uint64_t)_mm_extract_epi64(x, 0));
}
class multiply {
  const __m128i vp;
  __m128i reduce(__m128i ll, __m128i hh, __m128i mm) const {
    mm ^= _mm_clmulepi64_si128(hh, vp, 0x10);
    hh ^= _mm_shuffle_epi32(mm, _MM_SHUFFLE(1, 0, 3, 2));
    return ll ^ hh ^ _mm_clmulepi64_si128(hh, vp, 0x11);
  }
  __m128i reduce(__m128i ll, __m128i hh) const {
    __m128i mm = _mm_clmulepi64_si128(hh, vp, 0x10);
    hh ^= _mm_shuffle_epi32(mm, _MM_SHUFFLE(1, 0, 3, 2));
    return ll ^ hh ^ _mm_clmulepi64_si128(hh, vp, 0x11);
  }

 public:
  multiply(uint64_t p) : vp(_mm_set_epi64x(p, 0)) {}
  __m128i mul(__m128i a, __m128i b) const {
    return m1(mulp1(a, b));
  }
  __m128i pow(__m128i a, __uint128_t e) const {
    if (!e) {
      return _mm_set_epi64x(0x8000000000000000ULL, 0);
    }
    while (!(e & 1)) {
      a = sq(a);
      e >>= 1;
    }
    __m128i res = a;
    while (e >>= 1) {
      a = sq(a);
      if (e & 1)
        res = mul(res, a);
    }
    return res;
  }
  __m128i mulp1(__m128i a, __m128i b) const {
    __m128i ll = _mm_clmulepi64_si128(a, b, 0x11);
    __m128i hh = _mm_clmulepi64_si128(a, b, 0x00);
    __m128i lh = _mm_unpackhi_epi64(a, b) ^ _mm_unpacklo_epi64(a, b);
    __m128i mm = _mm_clmulepi64_si128(lh, lh, 0x01);
    lh = ll ^ hh;
    return reduce(ll, hh, mm ^ lh);
  }
  __m128i sq(__m128i a) const {
    return m1(sqp1(a));
  }
  __m128i sqp1(__m128i a) const {
    __m128i ll = _mm_clmulepi64_si128(a, a, 0x11);
    __m128i hh = _mm_clmulepi64_si128(a, a, 0x00);
    return reduce(ll, hh);
  }
  __m128i m1(__m128i a) const {
    /*
    alignas(16) __uint128_t res;
    _mm_store_si128((__m128i*)&res, a);
    res = res << 1 | res >> 127;
    if (res & 1) {
      res ^= (__uint128_t)p << 64;
    }
    return _mm_load_si128((__m128i*)&res);/*/
    __m128i b = _mm_srli_epi64(a, 63);
    __m128i m = _mm_srai_epi32(_mm_unpackhi_epi32(a, a), 31) & vp;  // these are all equivalent
    //__m128i m = _mm_srai_epi32(_mm_shuffle_epi32(a, _MM_SHUFFLE(3, 3, 3, 3)), 31) & vp;
    //__m128i m = _mm_sub_epi64(_mm_setzero_si128(), b) & vp;
    //__m128i m = _mm_srai_epi64(a, 64) & vp; // only with AVX512VL
    a = _mm_add_epi64(a, a) | _mm_shuffle_epi32(b, _MM_SHUFFLE(1, 0, 3, 2));
    return a ^ m;
    //*/
  }
  __m128i p(__m128i a, char i) const {
    const __m128i mask = _mm_set_epi64x(0, 1);
    do {  //*
      __m128i c = _mm_slli_epi64(a, 63);
      __m128i m = _mm_srai_epi32(_mm_shuffle_epi32(c, _MM_SHUFFLE(1, 1, 0, 0)), 31) & vp;
      a ^= m;
      __m128i b = _mm_srli_epi64(a, 1);
      a = b | _mm_shuffle_epi32(_mm_slli_epi64(a, 63), _MM_SHUFFLE(1, 0, 3, 2)); /*/
      //a ^= vp & _mm_set_epi64x(-(_mm_extract_epi64(a, 0) & 1), 0);
      //a = _mm_srli_epi64(a, 1) | _mm_shuffle_epi32(_mm_slli_epi64(a, 63), _MM_SHUFFLE(1, 0, 3, 2));
      //*/
    } while (--i > 0);
    return a;
  }
  __m128i p64(__m128i a) const {
    __m128i mult = _mm_clmulepi64_si128(a, vp, 0x10);
    return mult ^ _mm_shuffle_epi32(a, _MM_SHUFFLE(1, 0, 3, 2));
  }
};
__uint128_t mul(__uint128_t a, __uint128_t b, uint64_t p) {
  __uint128_t res = 0;
  do {
    if (b & x0)
      res ^= a;
    if (a & xn_1)
      a ^= (__uint128_t)p << 64;
    a = a >> 1 | a << 127;
    b <<= 1;
  } while (b);
  return res;
}
/*
int main() {
  alignas(16) __uint128_t a = (__uint128_t)0xfedcba9876543210UL << 64 | 0x5555555555555555UL;
  alignas(16) __uint128_t b = (__uint128_t)0xccccccccccccccccUL << 64 | 0x0123456789abcdefUL;
  uint64_t p = 0x8b5b159d0276d09dULL;
  __uint128_t expected = mul(a, b, p);
  print128(expected);
  expected = mul(a, a, p);
  print128(expected);
  multiply m(p);
  __m128i va = _mm_load_si128((__m128i*)&a);
  __m128i vb = _mm_load_si128((__m128i*)&b);
  __m128i ll = m.mul(va, vb);
  alignas(16) __uint128_t res;
  _mm_store_si128((__m128i*)&res, ll);
  print128(res);
  ll = m.sq(va);
  _mm_store_si128((__m128i*)&res, ll);
  print128(res);
}
*/