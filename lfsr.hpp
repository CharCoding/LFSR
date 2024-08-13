#pragma once

#include <immintrin.h>

#include <bit>
#include <cstddef>
#include <cstdint>
#include <type_traits>
#include <unordered_map>
#include <vector>
// big endian => x^127 is MSB, x^0 is LSB, shifts to the left
// little endian => x^0 is MSB, x^127 is LSB, shifts to the right
// hopefully this isn't a "Hey electrons should have positive charge" thing
constexpr uint64_t factors[] = {3, 5, 17, 257, 65537,
                                641, 6700417, 274177, 67280421310721};
// u8: 3,5,17; u16: 257; u32: 65537; u64: 641 6700417; u128: 274177 67280421310721
/*
MULTS + 1 = Actual chain length
641 is optimal with exp by squaring (11 == 11)
addition chain for 6700417: (27 < 32)
2 4 8 16 32 64 128 256 512 513 1026 2052 3078 3079 6158 12316 24632 49264 98528
197056 394112 788224 1576448 3152896 6305792 6699904 6700417
addition chain for 274177: (22 < 24)
2 4 8 16 32 64 128 256 512 768 1536 2304 4608 9216 13824 16128 32256 64512
129024 258048 274176 274177
addition chain for 67280421310721: (56 < 65)
2 4 8 16 17 34 51 102 153 306 459 918 1836 3672 7344 7803 15606 31212 31314
62628 125256 250512 501024 1002048 2004096 4008192 8016384 16032768 32065536
64131072 128262144 128324772 256649544 513299088 1026598176 2053196352
2053227666 4106455332 8212910664 8212941978 16425883956 16425884109 32851768218
65703536436 131407072872 262814145744 262814145745 525628291490 1051256582980
2102513165960 4205026331920 8410052663840 16820105327680 33640210655360
67280421310720 67280421310721
641 AND 6700417:
2 4 6 12 18 19 25 44 88 176 352 528 616 641 1282 2564 5128 7692 12820 25640 51280 102560 205120 410240 417932 835864 1671728 3343456 6686912 6699732 6700373 6700417 (32 <= 32)
274177 AND 67280421310721:
2 4 6 12 18 19 25 50 100 119 144 288 576 720 839 1678 2517 2661 5322 6161 8822 14983 23805 47610 62593 125186 250372 274177 548354 1096708 1370885 2467593 4935186 9870372 19740744 29611116 49351860 98703720 128314836 256629672 513259344 1026518688 2053037376 4106074752 8212149504 16424299008 32848598016 65697196032 131394392064 262788784128 525577568256 525626920116 525628291001 1051256582002 2102513164004 4205026328008 8410052656016 16820105312032 33640210624064 67280421248128 67280421310721 (61 <= 65)
P9 = 67280421310721 * (2^64 - 1) = (262814145745 * 2^8 + 1) * (2^64 - 1)
P8 = 274177 * (2^64 - 1) = (1071 * 2^8 + 1) * (2^64 - 1)
P7 = (2^32 - 1) * 641 * (2^64 + 1) = (5 * 2^7 + 1) * (2^32 - 1) * (2^64 + 1)
P6 = (2^32 - 1) * 6700417 * (2^64 + 1) = (52347 * 2^7 + 1) * (2^32 - 1) * (2^64 + 1)
P5 = (2^16 - 1) * (2^32 + 1) * (2^64 + 1)
P4 = (2^8 - 1) * (2^16 + 1) * (2^32 + 1) * (2^64 + 1)
P3 = (2^4 - 1) * (2^8 + 1) * (2^16 + 1) * (2^32 + 1) * (2^64 + 1)
P2 = (2^2 - 1) * (2^4 + 1) * (2^8 + 1) * (2^16 + 1) * (2^32 + 1) * (2^64 + 1)
P1 = (2^2 + 1) * (2^4 + 1) * (2^8 + 1) * (2^16 + 1) * (2^32 + 1) * (2^64 + 1)
P0 = (2^128 - 1)

SUMMATION
-1 (known)
1 (known)
2 = 2^1 (known)
3 (known)
...
128 (known)
We can only "cache" the first part of the multiplication when the base is still "2"
The remaining parts need to be built with continued fractions e.g. let y = x^(67280421310721), y^(2^64 - 1) = ? 1
But this is against the goal of caching the largest common factor
Cache:
2^64 ± 1 (66 mults)
L(0x3, 0xf, 0xff, 0xffff, 0xffffffff) = L(0xffffffff)
2^64 - 1 (69 mults, all previous mults are cached)
[2^64 - 1] + 22 + 56 (possible to cache between 274177 and 67280421310721)
[2^32 - 1] + 65 + 11 + 27 (possible to cache between 641 and 6700417)
[2^16 - 1] + 33 + 65
[2^8 - 1] + 17 + 33 + 65
[2^4 - 1] + 9 + 17 + 33 + 65
[2^2 - 1] + 5 + 9 + 17 + 33 + 65
3 + 5 + 9 + 17 + 33 + 65
*/
template <typename T>
constexpr unsigned char parity(T x) {
  static_assert(std::is_integral<T>::value, "T must be an integral type");
  return std::popcount<T>(x) & 1;
}
template <typename T>
constexpr T reverse_bits(T x) {
  static_assert(std::is_integral<T>::value, "T must be an integral type");
  if constexpr (sizeof(T) == 16) {
    const uint64_t hi = x >> 64, lo = (uint64_t)x;
    return reverse_bits(lo) << 64 | reverse_bits(hi);
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
template <typename T>
constexpr T evil_number_ordered(T x) { return (x << 1) | (T)parity<T>(x); }
template <typename T>
constexpr T evil_number_unordered(T x) { return x ^ std::rotl<T>(x, 1); }
namespace be {
template <typename T>
class lfsr {
  T state;
  const T polynomial;
  const T inv2;
  static constexpr T x0 = (T)1;
  static constexpr T xn_1 = (T)1 << (sizeof(T) * 8 - 1);
  // unordered_map<T, T> cache;
  vector<T> cache;
  static T multiply(T a, T b) {
    T r = 0;
    while (a && b) {
      if (b & x0) r ^= a;
      if (a & xn_1)
        a = a << 1 ^ polynomial;
      else
        a = a << 1;
      b >>= 1;
    }
    return r;
  }
  static T power(T b, T e) {
    T r = 1;
    while (e) {  // note that it doesn't make sense to make e follow endianness
      if (e & 1) r = multiply(r, b);
      b = multiply(b, b);
      e >>= 1;
    }
    return r;
  }
  static T powerOpt(T b, T e, T p) {
    if (e ==
        0) {  // unforunately 0 is the only "safe" value since -1 gets casted to
              // 2^(sizeof(T) * 8) - 1 which is a reasonable exponent
      return inv2;
    }
    if (e > 0 && e < sizeof(T) * 8) {
      return 1 << e;
    }
    if (e == sizeof(T) * 8) {
      return polynomial;
    }
    const auto none = cache.cend();
    auto it = cache.find(e);
    if (it != none) {
      return it->second;
    }
    it = cache.find(p);
    auto it2 = cache.find(e - p);
    if (it != none && it2 != none) {
      T prod = multiply(it->second, it2->second);
      cache[e] = prod;
      return prod;
    }
  }
  static bool is_primitive(T polynomial) {
    T x = polynomial;
    T e2_64;
    T e2_32;
    T e2_16;
    T e2_8;
    T e2_4;
    T e2_2;
  }

 public:
  lfsr(T polynomial)
      : state(x0), polynomial(polynomial), inv2(xn_1 | polynomial >> 1) {}
  lfsr(T polynomial, T seed)
      : state(seed), polynomial(polynomial), inv2(xn_1 | polynomial >> 1) {
    if (!seed) state = x0;
  }
  T next() { return state = (state << 1) ^ (state & xn_1 ? polynomial : 0); }
  T step(T steps) {}
  le::lfsr<T> to_le() {
    return le::lfsr<T>(reverse_bits(polynomial), reverse_bits(state));
  }
};
};  // namespace be
namespace le {
template <typename T>
class lfsr {
  T state;
  const T polynomial;
  static constexpr T x0 = (T)1 << (sizeof(T) * 8 - 1);
  static constexpr T xn_1 = (T)1;

 public:
  lfsr(T polynomial) : state(x0), polynomial(polynomial) {}
  lfsr(T polynomial, T seed) : state(seed), polynomial(polynomial) {
    if (!seed) state = x0;
  }
  T next() {
    state = (state >> 1) ^ (state & xn_1 ? polynomial : 0);
    return state;
  }
  T next_half() {}
  be::lfsr<T> to_be() {
    return be::lfsr<T>(reverse_bits(polynomial), reverse_bits(state));
  }
};
};  // namespace le