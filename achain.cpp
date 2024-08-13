#include <bit>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <unordered_map>
#include <vector>

using namespace std;

// The # of multiplications to compute x^n using the binary method
constexpr uint64_t mults(uint64_t n) { return bit_width(n) + popcount(n) - 2; }

namespace naive {
unordered_map<uint64_t, vector<uint64_t>> chains;

vector<uint64_t> cfchain(uint64_t);  // Addition chain using continued fractions

// Reference: http://www.numdam.org/item/JTNB_1994__6_1_21_0.pdf
vector<uint64_t> cfaux(uint64_t n, uint64_t d) {  // Chain({n, k}, total)
  uint64_t q = n / d, r = n % d;
  vector<uint64_t> vend = cfchain(q);
  for (auto& c : vend) c *= d;  // Multiply by d for "*" operation
  vector<uint64_t> vstart = (r > 1) ? cfaux(d, r) : cfchain(d);
  vstart.insert(vstart.cend(), vend.begin(), vend.end());
  if (r) vstart.push_back(n);  // Append n (= q * d + r) for "+" operation
  return vstart;
}

vector<uint64_t> cfchain(const uint64_t n) {  // MinChain(n, total)
  const auto it = chains.find(n);             // need better ways to store cached chains
  if (it != chains.cend()) return it->second;
  if ((n & (n - 3)) < 3) {  // return if n is a power of 2 plus {0, 1, 2}
    vector<uint64_t> v;
    v.reserve(bit_width(n) - ((n & 3) == 0));
    for (uint64_t i = 1; i < bit_width(n); i++) v.push_back(1 << i);
    if (n & 3) v.push_back(n);
    chains[n] = v;
    return v;
  }
  /* // Total strategy: O(n^2log^2(n))
  vector<uint64_t> min = {2};
  vector<uint64_t> chain = cfchain(n >> 1);
  for (auto& c : chain) c *= 2;
  min.insert(min.cend(), chain.begin(), chain.end());
  if (n & 1) min.push_back(n);
  for (uint64_t k = 3; k < n - 1; ++k) {
    chain = cfaux(n, k);
    if (chain.size() < min.size()) min = chain;
  }
  /*/
  // Dyadic strategy: O(nlog^3(n))
  uint64_t k = n >> 1;
  vector<uint64_t> min = cfchain(k);  // q = 2, r = n & 1
  min.push_back(n & ~1);
  if (n & 1) min.push_back(n);
  while ((k >>= 1) > 1) {
    vector<uint64_t> chain = cfaux(n, k);
    if (chain.size() < min.size()) min = chain;
    chain = cfaux(n, k ^ k >> 1);
    if (chain.size() < min.size()) min = chain;
    chain = cfaux(n, n / k);
    if (chain.size() < min.size()) min = chain;
    chain = cfaux(n, n / (k + 1));
    if (chain.size() < min.size()) min = chain;
  }
  //*/
  chains[n] = min;
  return min;
}
};  // namespace naive
namespace optimized {
vector<uint64_t> storage = {3, 5, 6, 7};
unordered_map<uint64_t, uint64_t> chains;
uint64_t cfchain(uint64_t);
vector<uint64_t> cfexpand(uint64_t n, uint64_t d) {
  uint64_t q = n / d, r = n % d;
  vector<uint64_t> vstart;
  if (r > 1) {
    vstart = cfexpand(d, r);
  } else if (d & d - 1) {
    uint64_t istart = cfchain(d);
    do {
      vstart.push_back(storage[istart]);
    } while (storage[istart++] != d);
  }

  uint64_t iend;
  if (q & q - 1) {
    iend = cfchain(q);
    if (vstart.size()) {
      do {
        vstart.push_back(vstart.back() * 2);
      } while (vstart.back() * 2 < storage[iend] * d);
    }
    do {
      vstart.push_back(storage[iend] * d);
    } while (storage[iend++] != q);
  } else if (vstart.size()) {
    iend = 2;
    while (iend <= q) {
      vstart.push_back(d *= 2);
      iend *= 2;
    }
  }
  if (r) {
    vstart.push_back(n);
  }
  return vstart;
}
uint64_t cfchain(const uint64_t n) {
  const auto it = chains.find(n);
  if (it != chains.cend()) return it->second;
  uint64_t start = storage.size();
  switch (popcount(n)) {
    case 3:
      storage.push_back(n & n - 1);
      [[fallthrough]];
    case 2:
      chains[n] = start;
      storage.push_back(n);
      return start;
  }
  const uint64_t half = n >> 1;
  uint64_t len = cfchain(half);
  vector<uint64_t> min;
  do {
    min.push_back(storage[len] * 2);
  } while (storage[len++] != half);
  if (n & 1) {
    min.push_back(n);
  }
  len = min.size() + bit_width(min.front());
  for (uint64_t k = 3; k < n - 1; ++k) {
    vector<uint64_t> chain = cfexpand(n, k);
    uint64_t clen = chain.size() + bit_width(chain.front());
    if (clen < len) {
      min = chain;
      len = clen;
    }
  }
  chains[n] = start = storage.size();
  storage.insert(storage.cend(), min.begin(), min.end());
  return start;
}
};  // namespace optimized
// Usage: ./achain <n> [d]
int main(int argc, char* argv[]) {
  if(argc == 1) {
    cout << "Usage: ./achain <n> [d]" << endl;
    return 1;
  }
  uint64_t n = strtoul(argv[1], NULL, 10);
  uint64_t d;
  if(argc == 3) {
    d = strtoul(argv[2], NULL, 10);
    naive::chains[1] = {};   // note that we don't include 1 in the chain
    naive::chains[2] = {2};  // otherwise we need to exclude it in every "*" operation
    naive::chains[3] = {2, 3};
    vector<uint64_t> min = naive::cfaux(n, d);
  }
  if(n > 16777219) {
    naive::chains[1] = {};   // note that we don't include 1 in the chain
    naive::chains[2] = {2};  // otherwise we need to exclude it in every "*" operation
    naive::chains[3] = {2, 3};
    vector<uint64_t> min = naive::cfchain(n);
    for (const auto step : min) cout << step << ' ';
    cout << '(' << min.size() << " <= " << mults(n) << ")\n";
    cout << "Cached " << naive::chains.size() << " chains\n";
  }
  /* // naive
  naive::chains[1] = {};   // note that we don't include 1 in the chain
  naive::chains[2] = {2};  // otherwise we need to exclude it in every "*" operation
  naive::chains[3] = {2, 3};
  vector<uint64_t> min = naive::cfchain(n);
  for (const auto step : min) cout << step << ' ';
  cout << '(' << min.size() << " <= " << mults(n) << ")\n";
  cout << "Cached " << naive::chains.size() << " chains\n";
  /*/
  // optimized
  optimized::chains[3] = 0;
  optimized::chains[5] = 1;
  optimized::chains[6] = optimized::chains[7] = 2;
  if (n <= 2) {
    cout << "Invalid input, n must be >= 2\n";
    return 1;
  }
  uint64_t start = (n & n - 1) ? optimized::cfchain(n) : -1UL;
  uint64_t len = 1;
  uint64_t step1 = (start == -1UL) ? n : optimized::storage[start];
  uint64_t doubling = 2;
  do {
    cout << doubling << ' ';
    ++len;
  } while ((doubling *= 2) <= step1);
  cout << step1;
  while (step1 < n) {
    step1 = optimized::storage[++start];
    ++len;
    cout << ' ' << step1;
  }
  cout << " (" << len << " <= " << mults(n) << ")\nCache size: " << optimized::storage.size() << ", chains: " << optimized::chains.size() << '\n';
  //*/
}