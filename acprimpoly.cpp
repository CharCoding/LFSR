#include <immintrin.h>
// compile with -O3 -msse4.1 -mpclmul [-DBE] ; (default is LE)
#include <bit>
#include <chrono>
#include <cstdint>
#include <cstdio>
template <typename T>
constexpr T reverse_bits(T x) {
  static_assert(std::is_integral<T>::value, "T must be an integral type");
  if constexpr (sizeof(T) == 16) {
    const uint64_t hi = x >> 64, lo = (uint64_t)x;
    return (T)reverse_bits(lo) << 64 | reverse_bits(hi);
  } else {
    T mask = ~(T)0;
    unsigned char s = sizeof(T) << 2;
    do {
      mask ^= mask << s;
      x = ((x >> s) & mask) | ((x << s) & ~mask);
    } while (s >>= 1);
    return x;
  }
}
// print functions do not differentiate between BE and LE
void print128(__uint128_t x) {
  printf("%016lx %016lx\n", (uint64_t)(x >> 64), (uint64_t)x);
}
// always prints [MSB:LSB]
void printm128(__m128i x) {
  printf("%016lx %016lx ", (uint64_t)_mm_extract_epi64(x, 1),
         (uint64_t)_mm_extract_epi64(x, 0));
}
#ifdef BE
constexpr __uint128_t xn_1 = (__uint128_t)1ULL << 127;  // MSB represents x^127
constexpr __uint128_t x0 = 1ULL;                        // LSB represents x^0
constexpr __uint128_t x1 = 2ULL;                        // shifts to the left
#else
constexpr __uint128_t xn_1 = 1;                       // LSB represents x^127
constexpr __uint128_t x0 = (__uint128_t)1ULL << 127;  // MSB represents x^0
constexpr __uint128_t x1 = (__uint128_t)1ULL << 126;  // shifts to the right
#endif
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
// multiplies 128-bit polynomials in GF(2^128), reduced by x^128 + p(x) + 1
// p(x) is a polynomial of degree at most 64 with no constant term
constexpr __uint128_t mul(__uint128_t a, __uint128_t b, uint64_t p) {
  __uint128_t res = 0;
  do {  // since a and b are powers of x and poly has a constant term of 1, they can never be 0
    if (b & x0)
      res ^= a;
    if (a & xn_1)  // invariant: a != 0; a will remain non-zero throughout the multiplication
#ifdef BE
      a ^= p;
    a = a << 1 | a >> 127;
    b >>= 1;
#else
      a ^= (__uint128_t)p << 64;
    a = a >> 1 | a << 127;
    b <<= 1;
#endif
  } while (b);  // b is not zero on the first iteration
  return res;
}

__m128i clreduce(__m128i lo, __m128i hi, __m128i vp) {
  /* reference impl.
  __m128i temp = _mm_clmulepi64_si128(hi, vp, 0x01);                   // temp[127:0] = hi[127:64] * vp[63:0]
  hi = _mm_xor_si128(hi, _mm_srli_si128(_mm_xor_si128(temp, hi), 8));  // hi[63:0] ^= temp[127:64] ^ hi[127:64]
  lo = _mm_xor_si128(lo, _mm_slli_si128(temp, 8));                     // lo[127:64] ^= temp[63:0]
  temp = _mm_clmulepi64_si128(hi, vp, 0x00);                           // temp[127:0] = hi[63:0] * vp[63:0]
  lo = _mm_xor_si128(lo, temp);                                        // lo[127:0] ^= temp[127:0]
  lo = _mm_xor_si128(lo, _mm_slli_si128(hi, 8));                       // lo[127:64] ^= hi[63:0]
  return lo;
  /*/
  // wow, saves 1 single instruction, so worthwhile...
#ifdef BE
  __m128i temp = _mm_clmulepi64_si128(hi, vp, 0x01);  // temp[127:0] = hi[127:64] * vp[63:0]
  temp = _mm_xor_si128(temp, hi);
  hi = _mm_xor_si128(hi, _mm_srli_si128(temp, 8));         // hi[63:0] ^= temp[127:64] ^ hi[127:64]
  lo = _mm_xor_si128(lo, _mm_unpacklo_epi64(temp, temp));  // lo[127:64] ^= temp[63:0]
  hi = _mm_clmulepi64_si128(hi, vp, 0x00);                 // temp[127:0] = hi[63:0] * vp[63:0]
  lo = _mm_xor_si128(lo, temp);                            // lo[127:0] ^= temp[127:0]
  lo = _mm_xor_si128(lo, hi);                              // lo[127:64] ^= hi[63:0]
#else
  __m128i temp = _mm_clmulepi64_si128(hi, vp, 0x00);
  hi ^= _mm_shuffle_epi32(temp, _MM_SHUFFLE(1, 0, 3, 2));
  lo ^= hi;
  temp = _mm_clmulepi64_si128(hi, vp, 0x01);
  lo ^= temp;
#endif
  return lo;
  //*/
}

__uint128_t clmul(__uint128_t a, __uint128_t b, uint64_t p) {
  __m128i va = _mm_set_epi64x((uint64_t)(a >> 64), (uint64_t)a);
  __m128i vb = _mm_set_epi64x((uint64_t)(b >> 64), (uint64_t)b);
#ifdef BE
  __m128i vp = _mm_set_epi64x(0, p << 1 | 1);                       // BE: x^64 set to 1, force X^0 to be set
  __m128i lo = _mm_clmulepi64_si128(va, vb, 0x00);                  // lo[127:0] = va[63:0] * vb[63:0]
  __m128i mid = _mm_xor_si128(_mm_clmulepi64_si128(va, vb, 0x01),   // mid[127:0] = va[63:0] * vb[127:64]
                              _mm_clmulepi64_si128(va, vb, 0x10));  // mid[127:0] ^= va[127:64] * vb[63:0]
  __m128i hi = _mm_clmulepi64_si128(va, vb, 0x11);                  // hi[127:0] = va[127:64] * vb[127:64]
  hi = _mm_xor_si128(hi, _mm_srli_si128(mid, 8));                   // hi[63:0] ^= mid[127:64]
  lo = _mm_xor_si128(lo, _mm_slli_si128(mid, 8));                   // lo[127:64] ^= mid[63:0]
  /*
  mid = _mm_clmulepi64_si128(hi, vp, 0x01);                           // temp[127:0] = hi[127:64] * vp[63:0]
  hi = _mm_xor_si128(hi, _mm_srli_si128(_mm_xor_si128(mid, hi), 8));  // hi[63:0] ^= temp[127:64] ^ hi[127:64]
  lo = _mm_xor_si128(lo, _mm_slli_si128(mid, 8));                     // lo[127:64] ^= temp[63:0]
  mid = _mm_clmulepi64_si128(hi, vp, 0x00);                           // temp[127:0] = hi[63:0] * vp[63:0]
  lo = _mm_xor_si128(lo, mid);                                        // lo[127:0] ^= temp[127:0]
  lo = _mm_xor_si128(lo, _mm_slli_si128(hi, 8));                      // lo[127:64] ^= hi[63:0]
  */

#else
  __m128i vp = _mm_set_epi64x(0, p);  // LE: everything is shifted to the right by 1 bit
  __m128i lo = _mm_clmulepi64_si128(va, vb, 0x11);
  __m128i mid = _mm_clmulepi64_si128(va, vb, 0x10) ^ _mm_clmulepi64_si128(va, vb, 0x01);
  __m128i hi = _mm_clmulepi64_si128(va, vb, 0x00);
  hi ^= _mm_slli_si128(mid, 8);  // hi[127:64] ^= mid[63:0]
  lo ^= _mm_srli_si128(mid, 8);  // lo[63:0] ^= mid[127:64]
  // lo = lo << 1 | hi >> 127;
  // hi = hi << 1;
  //*
  alignas(16) __uint128_t h, l;
  _mm_store_si128((__m128i*)&h, hi);
  _mm_store_si128((__m128i*)&l, lo);
  l = l << 1 | h >> 127;
  h += h;
  hi = _mm_load_si128((__m128i*)&h);
  lo = _mm_load_si128((__m128i*)&l);
  //*/
#endif
  lo = clreduce(lo, hi, vp);
  alignas(16) __uint128_t res;
  _mm_store_si128((__m128i*)&res, lo);
  return res;
}

__uint128_t clsq(__uint128_t a, uint64_t p) {
  __m128i va = _mm_set_epi64x((uint64_t)(a >> 64), (uint64_t)a);
  __m128i lo = _mm_clmulepi64_si128(va, va, 0x00);
  __m128i hi = _mm_clmulepi64_si128(va, va, 0x11);
#ifdef BE
  __m128i vp = _mm_set_epi64x(0, p << 1 | 1);
#else
  __m128i vp = _mm_set_epi64x(0, p);
  alignas(16) __uint128_t h, l;
  _mm_store_si128((__m128i*)&h, lo);  // swap lo and hi due to endianness
  _mm_store_si128((__m128i*)&l, hi);
  l += l;  // h >> 127 == 0
  h += h;
  hi = _mm_load_si128((__m128i*)&h);
  lo = _mm_load_si128((__m128i*)&l);
#endif
  lo = clreduce(lo, hi, vp);
  alignas(16) __uint128_t res;
  _mm_store_si128((__m128i*)&res, lo);
  return res;
}

// raise a to the power of e, reduced by x^128 + p(x) + 1
// this uses the minimal number of multiplications using the binary method
__uint128_t pow(__uint128_t a, __uint128_t e, uint64_t p) {
  if (!e) return x0;
  while ((e & 1) == 0) {
    a = clsq(a, p);  // clmul(a, a, p);
    e >>= 1;
  }
  __uint128_t res = a;  // assign instead of multiplying by 1 to avoid unnecessary multiplication
  while (e >>= 1) {
    a = clsq(a, p);  // clmul(a, a, p);
    if (e & 1)
      res = clmul(res, a, p);
  }
  return res;
}

//* // error checking
bool assert128(__uint128_t x, __uint128_t y) {
  if (x == y) {
    printf("%016lx %016lx equal\n", (uint64_t)(x >> 64), (uint64_t)x);
    return true;
  }
  printf("%016lx %016lx != %016lx %016lx\n", (uint64_t)(x >> 64), (uint64_t)x, (uint64_t)(y >> 64), (uint64_t)y);
  return false;
}
/*/
#define assert128(x, y)  // skipped
//*/
// Check if x^128 + p(x) + 1 is a primitive polynomial under the given endianness
// BE: MSB of poly represents x^64, LSB represents x^1
// LE: LSB of poly represents x^64, MSB represents x^1
bool isPrimitive(const uint64_t poly) {
#ifdef BE
  __uint128_t xm1 = poly | xn_1;                  // mult. inverse of x
  __uint128_t xp1 = (__uint128_t)poly << 1 | x0;  // quicky calculate x^128; we need to get to x^(2^64)
#else
  __uint128_t xm1 = (__uint128_t)poly << 64 | xn_1;  // mult. inverse of x
  __uint128_t xp1 = (__uint128_t)poly << 63 | x0;    // quicky calculate x^128; we need to get to x^(2^64)
#endif
  assert128(mul(xm1, x1, poly), x0);
  {  // (57 more squarings)
    int i = 57;
    do
      xp1 = mul(xp1, xp1, poly);
    while (--i);
  }
  xm1 = mul(xp1, xm1, poly);  // x^(2^64-1): the only time we can multiply by x^-1 to speed up
  if (xp1 & xn_1)             // x^(2^64+1): the only time we can multiply by x to speed up
#ifdef BE
    xp1 ^= poly;
  xp1 = xp1 << 1 | xp1 >> 127;
#else
    xp1 ^= (__uint128_t)poly << 64;
  xp1 = xp1 >> 1 | xp1 << 127;
#endif
  /* Exponentiation by squaring: ~1800 multiplications
     With caching: ~700 multiplications
     Addition chain & 2^(2^x) + 1 ladder: 230 multiplications (at the cost of larger code size)
     One might be able to do better using purely addition chains, but that will take a lot of computing power to find
     It's also worth noting that the first 9 checks rarely fail; it's the last one that is most likely to fail
     So it would make sense to compute x^(2^128 - 1) first
     But the way I have it set up, going toward x^(2^128 - 1) naturally visits most of the products already
  */
  {  // find x^(2^64 - 1)(274177) and x^(2^64 - 1)(67280421310721) using addition chain exponentiation
    __uint128_t a = xm1;
    __uint128_t b = mul(a, a, poly);  // 2
    __uint128_t c = mul(b, b, poly);  // 4
    b = mul(c, b, poly);              // 6
    c = mul(b, b, poly);              // 12
    c = mul(c, b, poly);              // 18
    a = mul(c, a, poly);              // 19
    b = mul(a, b, poly);              // 25
    c = mul(b, b, poly);              // 50
    c = mul(c, c, poly);              // 100
    a = mul(c, a, poly);              // 119
    b = mul(a, b, poly);              // 144
    c = mul(b, b, poly);              // 288
    c = mul(c, c, poly);              // 576
    c = mul(c, b, poly);              // 720
    a = mul(c, a, poly);              // 839
    c = mul(a, a, poly);              // 1678
    c = mul(c, a, poly);              // 2517
    b = mul(c, b, poly);              // 2661
    c = mul(b, b, poly);              // 5322
    a = mul(c, a, poly);              // 6161
    b = mul(a, b, poly);              // 8822
    a = mul(b, a, poly);              // 14983
    b = mul(a, b, poly);              // 23805
    c = mul(b, b, poly);              // 47610
    a = mul(c, a, poly);              // 62593
    c = mul(a, a, poly);              // 125186
    c = mul(c, c, poly);              // 250372
    b = mul(c, b, poly);              // 274177
    assert128(b, pow(x1, products[8], poly));
    if (b == x0)
      return false;
    c = mul(b, b, poly);              // 548354
    c = mul(c, c, poly);              // 1096708
    b = mul(c, b, poly);              // 1370885
    c = mul(b, c, poly);              // 2467593
    c = mul(c, c, poly);              // 4935186
    c = mul(c, c, poly);              // 9870372
    __uint128_t d = mul(c, c, poly);  // 19740744
    c = mul(d, c, poly);              // 29611116
    d = mul(c, d, poly);              // 49351860
    __uint128_t e = mul(d, d, poly);  // 98703720
    c = mul(e, c, poly);              // 128314836
    c = mul(c, c, poly);              // 256629672
    c = mul(c, c, poly);              // 513259344
    c = mul(c, c, poly);              // 1026518688
    c = mul(c, c, poly);              // 2053037376
    c = mul(c, c, poly);              // 4106074752
    c = mul(c, c, poly);              // 8212149504
    c = mul(c, c, poly);              // 16424299008
    c = mul(c, c, poly);              // 32848598016
    c = mul(c, c, poly);              // 65697196032
    c = mul(c, c, poly);              // 131394392064
    c = mul(c, c, poly);              // 262788784128
    c = mul(c, c, poly);              // 525577568256
    d = mul(c, d, poly);              // 525626920116
    b = mul(d, b, poly);              // 525628291001
    b = mul(b, b, poly);              // 1051256582002
    b = mul(b, b, poly);              // 2102513164004
    b = mul(b, b, poly);              // 4205026328008
    b = mul(b, b, poly);              // 8410052656016
    b = mul(b, b, poly);              // 16820105312032
    b = mul(b, b, poly);              // 33640210624064
    b = mul(b, b, poly);              // 67280421248128
    a = mul(b, a, poly);              // 67280421310721
    assert128(a, pow(x1, products[7], poly));
    if (a == x0)
      return false;
  }
  {  // find x^(2^64 + 1)(2^32 ± 1)
    __uint128_t a = xp1;
    __uint128_t b = mul(a, a, poly);  // 2
    a = mul(b, a, poly);              // 3
    __uint128_t c = mul(a, a, poly);  // 6
    c = mul(c, c, poly);              // 12
    a = mul(c, a, poly);              // 15
    c = mul(a, a, poly);              // 30
    c = mul(c, c, poly);              // 60
    c = mul(c, c, poly);              // 120
    c = mul(c, c, poly);              // 240
    a = mul(c, a, poly);              // 255
    c = mul(a, a, poly);              // 510
    c = mul(c, c, poly);              // 1020
    c = mul(c, c, poly);              // 2040
    c = mul(c, c, poly);              // 4080
    c = mul(c, c, poly);              // 8160
    c = mul(c, c, poly);              // 16320
    c = mul(c, c, poly);              // 32640
    c = mul(c, c, poly);              // 65280
    a = mul(c, a, poly);              // 65535
    c = mul(a, a, poly);              // 131070
    c = mul(c, c, poly);              // 262140
    c = mul(c, c, poly);              // 524280
    c = mul(c, c, poly);              // 1048560
    c = mul(c, c, poly);              // 2097120
    c = mul(c, c, poly);              // 4194240
    c = mul(c, c, poly);              // 8388480
    c = mul(c, c, poly);              // 16776960
    c = mul(c, c, poly);              // 33553920
    c = mul(c, c, poly);              // 67107840
    c = mul(c, c, poly);              // 134215680
    c = mul(c, c, poly);              // 268431360
    c = mul(c, c, poly);              // 536862720
    c = mul(c, c, poly);              // 1073725440
    c = mul(c, c, poly);              // 2147450880
    c = mul(c, c, poly);              // 4294901760
    xm1 = mul(c, a, poly);            // 4294967295
    xp1 = mul(xm1, b, poly);          // 4294967297
  }
  {  // find x^(2^64 + 1)(2^32 - 1)(641) and x^(2^64 + 1)(2^32 - 1)(6700417)
    __uint128_t a = xm1;
    __uint128_t b = mul(a, a, poly);  // 2
    __uint128_t c = mul(b, b, poly);  // 4
    b = mul(c, b, poly);              // 6
    c = mul(b, b, poly);              // 12
    c = mul(c, b, poly);              // 18
    a = mul(c, a, poly);              // 19
    b = mul(a, b, poly);              // 25
    a = mul(b, a, poly);              // 44
    c = mul(a, a, poly);              // 88
    __uint128_t d = mul(c, c, poly);  // 176
    __uint128_t e = mul(d, d, poly);  // 352
    d = mul(e, d, poly);              // 528
    c = mul(d, c, poly);              // 616
    b = mul(c, b, poly);              // 641
    assert128(b, pow(x1, products[6], poly));
    if (b == x0)
      return false;
    c = mul(b, b, poly);  // 1282
    c = mul(c, c, poly);  // 2564
    d = mul(c, c, poly);  // 5128
    c = mul(d, c, poly);  // 7692
    d = mul(c, d, poly);  // 12820
    e = mul(d, d, poly);  // 25640
    e = mul(e, e, poly);  // 51280
    e = mul(e, e, poly);  // 102560
    e = mul(e, e, poly);  // 205120
    e = mul(e, e, poly);  // 410240
    c = mul(e, c, poly);  // 417932
    c = mul(c, c, poly);  // 835864
    c = mul(c, c, poly);  // 1671728
    c = mul(c, c, poly);  // 3343456
    c = mul(c, c, poly);  // 6686912
    d = mul(c, d, poly);  // 6699732
    b = mul(d, b, poly);  // 6700373
    a = mul(b, a, poly);  // 6700417
    assert128(a, pow(x1, products[5], poly));
    if (a == x0)
      return false;
  }
  {  // find x^(2^64 + 1)(2^32 + 1)(2^16 ± 1)
    __uint128_t a = xp1;
    __uint128_t b = mul(a, a, poly);  // 2
    a = mul(b, a, poly);              // 3
    __uint128_t c = mul(a, a, poly);  // 6
    c = mul(c, c, poly);              // 12
    a = mul(c, a, poly);              // 15
    c = mul(a, a, poly);              // 30
    c = mul(c, c, poly);              // 60
    c = mul(c, c, poly);              // 120
    c = mul(c, c, poly);              // 240
    a = mul(c, a, poly);              // 255
    c = mul(a, a, poly);              // 510
    c = mul(c, c, poly);              // 1020
    c = mul(c, c, poly);              // 2040
    c = mul(c, c, poly);              // 4080
    c = mul(c, c, poly);              // 8160
    c = mul(c, c, poly);              // 16320
    c = mul(c, c, poly);              // 32640
    c = mul(c, c, poly);              // 65280
    xm1 = mul(c, a, poly);            // 65535
    assert128(xm1, pow(x1, products[4], poly));
    if (xm1 == x0)
      return false;
    xp1 = mul(xm1, b, poly);  // 65537
  }
  {  // find x^(2^64 + 1)(2^32 + 1)(2^16 + 1)(2^8 ± 1)
    __uint128_t a = xp1;
    __uint128_t b = mul(a, a, poly);  // 2
    a = mul(b, a, poly);              // 3
    __uint128_t c = mul(a, a, poly);  // 6
    c = mul(c, c, poly);              // 12
    a = mul(c, a, poly);              // 15
    c = mul(a, a, poly);              // 30
    c = mul(c, c, poly);              // 60
    c = mul(c, c, poly);              // 120
    c = mul(c, c, poly);              // 240
    xm1 = mul(c, a, poly);            // 255
    assert128(xm1, pow(x1, products[3], poly));
    if (xm1 == x0)
      return false;
    xp1 = mul(xm1, b, poly);  // 257
  }
  {  // find x^(2^64 + 1)(2^32 + 1)(2^16 + 1)(2^8 + 1)(2^4 ± 1)
    __uint128_t a = xp1;
    __uint128_t b = mul(a, a, poly);  // 2
    a = mul(b, a, poly);              // 3
    __uint128_t c = mul(a, a, poly);  // 6
    c = mul(c, c, poly);              // 12
    xm1 = mul(c, a, poly);            // 15
    assert128(xm1, pow(x1, products[2], poly));
    if (xm1 == x0)
      return false;
    xp1 = mul(xm1, b, poly);  // 17
  }
  {  // find x^(2^64 + 1)(2^32 + 1)(2^16 + 1)(2^8 + 1)(2^4 + 1)(2^2 ± 1)
    __uint128_t a = xp1;
    __uint128_t b = mul(a, a, poly);  // 2
    a = mul(b, a, poly);              // 3
    assert128(a, pow(x1, products[1], poly));
    if (a == x0)
      return false;
    b = mul(a, b, poly);  // 5
    assert128(b, pow(x1, products[0], poly));
    if (b == x0)
      return false;
    a = mul(b, b, poly);  // 10
    a = mul(a, b, poly);  // 15
    assert128(a, pow(x1, ~(__uint128_t)(0), poly));
    return a == x0;  // at this point, poly is primitive iff x^(2^128 - 1) == x^0
  }
}
// prng
constexpr uint64_t lcg_next(uint64_t lcg_state) {
  return lcg_state * 6364136223846793005ULL + 1ULL;
}

void generate_poly(uint64_t lcg_state, char count) {
  while (count > 0) {
    lcg_state = lcg_next(lcg_state);
    uint64_t px = lcg_state;  // we're generating x^64 ~ x^1; the constant term is always set to 1
    // px ^= (px << 1 | px >> 63) ^ 1;  // ensure an odd number of set bits; otherwise the polynomial is factorable by x + 1
    // note: we do not force x^64 to be set; but polynomials without x^64 term will not work in update64 in big endian
    // px |= 0x8000000000000001ULL;  // force x^64 to be set
    // px ^= px >> 1;
    px ^= (px << 1 | px >> 63);
    px ^= 2 - ((std::popcount(px & 0x4924924924924924ULL) |
                std::popcount(px & 0x2492492492492492ULL)) &
               1);  // avoid polynomials divisible by x + 1 and x^2 + x + 1
    if (isPrimitive(px)) {
      printf("%016lx\nx^128 + ", px);  // x^128 is implied
      for (int i = 63; i >= 0; --i)    // print algebraic expression of P(x)
#ifdef BE
        if (px & (1ULL << i))
#else
        if (px & (1ULL << (63 - i)))
#endif
          printf("x^%d + ", (i + 1));
      puts("1");
      --count;
    }
  }
}
// single step of a 128-bit LFSR
constexpr __uint128_t update(__uint128_t x, uint64_t p) {
  if (x & xn_1)
#ifdef BE
    x ^= p;
  return x << 1 | x >> 127;
#else
    x ^= (__uint128_t)p << 64;
  return x >> 1 | x << 127;
#endif
}
/* The update64 functions are not complete mirrors of each other
   the BE variant forces x^64 to be set, while the LE variant forces x^0 to be set
   Now, of course, x^0 must be set either way to be a primitive polynomial
   Which means the BE variant has one less usable bit
   It is also forced to use one additional SSE instruction (unless you know how to save it?)
*/
// 64 steps of a 128-bit LFSR
inline __m128i update64(__m128i state, uint64_t p) {
#ifdef BE
  __m128i taps = _mm_set_epi64x(0, p << 1 | 1);
  __m128i mult = _mm_clmulepi64_si128(state, taps, 0x01);
  __m128i cloned = _mm_shuffle_epi32(state, _MM_SHUFFLE(1, 0, 1, 0));
  __m128i zeroed = _mm_xor_si128(state, cloned);
  return _mm_xor_si128(zeroed, mult);
#else
  __m128i taps = _mm_set_epi64x(0, p);
  __m128i mult = _mm_clmulepi64_si128(state, taps, 0x00);
  __m128i swapped = _mm_shuffle_epi32(state, _MM_SHUFFLE(1, 0, 3, 2));
  return _mm_xor_si128(mult, swapped);
#endif
}

int main() {
  const uint64_t be1le1 = 0x8b5b159d0276d09dULL,
                 be1le0 = 0xf5fa1820020c2fd7ULL,
                 be0le1 = be1le0 << 1 | be1le0 >> 63,
                 be0le0 = be1le0 >> 1 | be1le0 << 63;
#ifdef BE
  const char* endianness = "not primitive in BE\n";
#else
  const char* endianness = "not primitive in LE\n";
#endif
  puts("isPrimitive check:");
  printf("%016lx is %s", be1le1, endianness + isPrimitive(be1le1) * 4);
  return 0;
  printf("%016lx is %s", be1le0, endianness + isPrimitive(be1le0) * 4);
  printf("%016lx is %s", be0le1, endianness + isPrimitive(be0le1) * 4);
  printf("%016lx is %s", be0le0, endianness + isPrimitive(be0le0) * 4);
  puts("LFSR check:");
  uint64_t poly = be0le1;
  __uint128_t lfsr_state = x0;
  __m128i lfsr_statem = _mm_set_epi64x((uint64_t)(lfsr_state >> 64), (uint64_t)lfsr_state);
  printm128(lfsr_statem);
  print128(lfsr_state);
  for (int i = 10; i--;) {
    for (int j = 64; j--;) {
      lfsr_state = update(lfsr_state, poly);
    }
    lfsr_statem = update64(lfsr_statem, poly);
    printm128(lfsr_statem);
    print128(lfsr_state);
  }
  if ((uint64_t)(lfsr_state >> 64) != _mm_extract_epi64(lfsr_statem, 1) || (uint64_t)lfsr_state != _mm_extract_epi64(lfsr_statem, 0))
    puts("LFSR check failed");
  else
    puts("LFSR check passed");
  //*
  puts("Testing clmul/clsq:");
  __uint128_t a = lfsr_state;
  __uint128_t b = (lfsr_state >> 64) * (lfsr_state & 0xFFFFFFFFFFFFFFFFULL);
  print128(a);
  print128(b);
  __uint128_t expected = mul(a, a, be0le1);
  __uint128_t actual = clmul(a, a, be0le1);
  printf("Expected: ");
  print128(expected);
  printf("Actual:   ");
  print128(actual);
  actual ^= expected;
  if (actual == 0)
    puts("clmul check passed");
  else {
    puts("clmul check failed");
    printf("Diff:     ");
    print128(actual);
  }
  expected = mul(b, b, be0le1);
  actual = clsq(b, be0le1);
  printf("Expected: ");
  print128(expected);
  printf("Actual:   ");
  print128(actual);
  actual ^= expected;
  if (actual == 0)
    puts("clsq check passed");
  else {
    puts("clsq check failed");
    printf("Diff:     ");
    print128(actual);
  }
  //*/
  puts("Generating polynomials:");
  generate_poly(std::chrono::high_resolution_clock::now().time_since_epoch().count(), 10);
  //*/
}