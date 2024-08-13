# 128-bit LFSR with update64 function
## Usage
```bash
g++ -O3 -std=c++20 -o achain achain.cpp
# compute the continued fraction addition chain for n
# should take less than 100 seconds; search more thoroughly for n < 2^24
./achain <n>
# compute the continued fraction addition chain for n that contains d, n > d
# Unfortunately, it can't find 3 or more targets simultaneously.
./achain <n> <d>
# acprimpoly just runs a few checks and generates 10 primitive polynomials of the form x^128 + p(x) + 1 in the given endianness
# The code is semi-documented for you to experiment with.
g++ -O3 -msse4.1 -mpclmul -o acprimpoly_le acprimpoly.cpp
./acprimpoly_le
g++ -DBE -O3 -msse4.1 -mpclmul -o acprimpoly_be acprimpoly.cpp
./acprimpoly_be
g++ -O3 -msse4.1 -mpclmul -o lfsr lfsr.cpp
# Run the LFSR128-64 prng (little-endian, outputs raw bytes to stdout)
./lfsr
```
Do not use `./lfsr` for cryptographic purposes. It is a pseudorandom number generator with known vulnerabilities.
## Motivation
LFSRs only generate 1 new bit per step, which is slow. We often want to generate multiples of a byte at once.  
Using a primitive polynomial of the form `P(x) = x^128 + p(x) + 1` where `p(x)` has degree at most 64, we can generate 64 bits at once using a carryless multiplication instruction:
```c
inline __m128i update64(__m128i state, uint64_t p) {
  __m128i taps = _mm_set_epi64x(0, p);
  __m128i mult = _mm_clmulepi64_si128(state, taps, 0x00);
  __m128i swapped = _mm_shuffle_epi32(state, _MM_SHUFFLE(1, 0, 3, 2));
  return _mm_xor_si128(mult, swapped);
}
```
Here, `state` holds the LFSR state (must be nonzero), and `p` represents the coefficients of x^1 (MSB) to x^64 (LSB).  
This representation will be referred as little-endian. The smallest power is written first (that is, if you write from left to right), and the whole register is shifted to the right after each step.

## Side quest 0: Finding primitive polynomials
LFSRs need the taps to be primitive polynomials in order to have a maximal period of 2^128 - 1.  
In order to find primitive polynomials of the form `P(x) = x^128 + p(x) + 1`, we need to check:
1. Raising the polynomial `x` to the power of 2^128 - 1 modulo `P(x)` and checking if it equals 1.
2. For each prime factor `f` of 2^128 - 1, raising the polynomial `x` to the `(2^128 - 1) / f`-th power modulo `P(x)` and checking if it does not equal 1.
The second step essentially checks if `x^d` does not equal 1 for every non-trivial divisor `d` of 2^128 - 1, which makes sure `x` is a generator of the multiplicative group of GF(2^128).  
A polynomial cannot be primitive in GF(2^n) if it has an even number of terms, since it would be divisible by `(x + 1)`. `x^128` and 1 are implied in `P(x)`, so `p(x)` must have an odd number of terms, or, equivalently, `p` must have an odd number of set bits. We can ensure this by doing `p ^= ~__builtin_popcountll(p) & 1` to get to the closest odious number, or `p ^= p << 1 | p >> 63 ^ 1` to get to an odious number in the fewest operations. This makes generating a primitive polynomial twice as likely.

## Side quest 1: Addition chain
Primitive polynomials are rare, and raising polynomials to large powers is slow, even using exponentiation by squaring. It takes around 1800 multiplications to compute these powers, and even if you cache and reuse `x^2`, `x^4`, ..., `x^(2^127)` you still need around 700 multiplications.  
A faster way to compute these powers is to use addition chains.

However, optimal addition chains themselves are generally hard to compute (it's NP-complete?).  
So instead of building one complicated chain to reach all 9 large powers, we'll concatenate smaller and simpler chains. For example, if chain A = {1, 2, ..., x} and chain B = {1, 2, ..., y}, we define A||B to be {1, 2, ..., x, 2x, ..., xy}. A||B has length |A| + |B| - 1.  
It's not too difficult to find the optimal addition chain for 2^(2^n) - 1: starting with a chain that ends in 2^(2^(n - 1)) - 1, double the last element if it has more set bits than trailing zeros; otherwise, add the element 2^(2^(n - 1)) - 1 to the last element.  
Since 2 is in every addition chain, we can also get to 2^(2^n) + 1 with one more addition.  
As a result, we can reach the following powers:
- 2^64 ± 1
- (2^64 + 1)(2^32 ± 1)
- (2^64 + 1)(2^32 + 1)(2^16 ± 1)
- (2^64 + 1)(2^32 + 1)(2^16 + 1)(2^8 ± 1)
- (2^64 + 1)(2^32 + 1)(2^16 + 1)(2^8 + 1)(2^4 ± 1)
- (2^64 + 1)(2^32 + 1)(2^16 + 1)(2^8 + 1)(2^4 + 1)(2^2 ± 1)
- (2^64 + 1)(2^32 + 1)(2^16 + 1)(2^8 + 1)(2^4 + 1)(2^2 + 1)(2^1 + 1) = 2^128 - 1
This covers 5 of the divisors and 2^128 - 1 with a near-optimal number of multiplications.

## Side quest 2: Fermat numbers aren't all prime
In order to reach the prime factors 641 × 6700417 = 2^32 + 1 and 274177 × 67280421310721 = 2^64 + 1, we need to use the continued fraction method for approximating addition chains. It is similar to finding the GCD:  
Let `d` be the smaller target and `n` be the larger. To reach both `d` and `n`, we'll first reach `d` through a step `r`, where `r = n mod d`, then we reach `n` by concatenating the chain for `d` and the chain for `n / d`, then adding `r` to the last element of the chain, which equals `n`.  
It works recursively, so now `d` is the larger target and `r` is the smaller target. As for computing the addition chain for `n / d`, if it is small enough, we can try each `r` in {2, ..., n / d - 2}, compute its continued fraction addition chain and pick the shortest chain. If not, we will only pick from `r = n / 2^k` for `1 < k < log2(n)`.
There may be better methods, but this is good enough. It saves 39 multiplications compared to using exponentiation by squaring.  
By concatenating these chains with the chains for (2^64 + 1)(2^32 - 1) and (2^64 - 1) respectively, we have reached all the remaining divisors.  
With a few more bithacks, only 230 multiplications are required overall.

## Side quest 3: Converting addition chains to instructions, minimizing the number of variables stored simultaneously
Given an addition chain, we can compute it by just storing the result of each step of the chain in a hash table, and looking them up when needed. However, often we only need to store a few of the intermediate results, and we can overwrite ones that are no longer needed.
Compilers probably know how to do this, but it's not obvious how to do it manually.  
1. Go through the addition chain from the start, recording the last index where each element is needed in a hash table. The hash table also contains the variable name that will be assigned to each element, which is initially empty.
2. Initialize a stack of variable names, containing the character 'b'
3. Assign 'a' to the first element (must be 1) of the chain.
4. Starting from the second element, for each element in the chain:
    1. Find the two previous elements ("summands") that sums to the current element.
    2. If the larger summand's last used index is the current index, push its variable name onto the stack.
    3. If the other summand is different from the first, and its last used index is the current index, push its variable name onto the stack.
    4. Assign the variable name at the top of the stack to the current element.
    5. Output the instruction "current := larger + other". Modify this expression as necessary for your programming language.
    6. If the stack size is 1, increment the variable name at the top of the stack. Otherwise, pop the top of the stack.

This algorithm would be O(n^2) in general due to searching for the pair of summands, but in our use case, we have the special property that the larger summand is always the immediate predecessor of the current element, and looking up the other summand takes constant time in a hash table. This kind of addition chain is called a "star chain". Star chains are not always optimal, but are easier to compute.

## Side quest 4: Implementing the LFSR in big-endian
Interpreting a register as a big-endian number is usually more intuitive, since the n-th bit represents the coefficient for `x^n`, but in this case it's the opposite.  
This is basically derived via trial and error.
```c
inline __m128i update64(__m128i state, uint64_t p) {
  __m128i taps = _mm_set_epi64x(0, p << 1 | 1);
  __m128i mult = _mm_clmulepi64_si128(state, taps, 0x01);
  __m128i cloned = _mm_shuffle_epi32(state, _MM_SHUFFLE(1, 0, 1, 0));
  __m128i zeroed = _mm_xor_si128(state, cloned);
  return _mm_xor_si128(zeroed, mult);
}
```
Note that the `x^64` term is forced to be set. There is no way to avoid this without also changing the shuffle and xor instructions. Then you have to do branching, which defeats the purpose of the LFSR.