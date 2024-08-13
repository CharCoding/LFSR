#include <cstdio>
#include <span>
#include <unordered_map>
#include <vector>

using namespace std;

typedef struct operation {
  char sum;
  char addend;
  char adder;
} operation;

vector<operation> generateChain(const span<unsigned long> chain) {
  unordered_map<unsigned long, pair<char, char>> ref;
  vector<char> stack = {'b'};
  vector<operation> ops;
  unsigned long prev = 1;
  for (size_t i = 1; i < chain.size(); ++i) {
    const unsigned long element = chain[i];
    const unsigned long diff = element - prev;
    ref[prev].first = i;
    if (diff != prev)
      ref[diff].first = i;
    prev = element;
  }
  prev = 1;
  ref[1].second = 'a';
  puts("a = x; // 1");
  for (size_t i = 1; i < chain.size(); ++i) {
    const unsigned long element = chain[i];
    const unsigned long diff = element - prev;
    if (ref[prev].first == i)
      stack.push_back(ref[prev].second);
    if (diff != prev && ref[diff].first == i)
      stack.push_back(ref[diff].second);
    const char var = stack.back();
    printf("%c = mul(%c, %c, poly); // %lu\n", var, ref[prev].second, ref[diff].second, element);
    ops.push_back({var, ref[prev].second, ref[diff].second});
    ref[element].second = var;
    if (stack.size() == 1)
      stack.front() = var + 1;
    else
      stack.pop_back();
    prev = element;
  }
  return ops;
}

bool verifyChain(const span<unsigned long> chain, const vector<operation> ops) {
  if (chain.size() != ops.size() + 1) {
    printf("Chain size %lu != ops size %lu + 1\n", chain.size(), ops.size());
    return false;
  }
  unsigned long values[10] = {0};
  values[0] = 1;
  for (size_t i = 0; i < ops.size(); ++i) {
    const operation op = ops[i];
    const unsigned long adder = values[op.adder - 'a'];
    const unsigned long addend = values[op.addend - 'a'];
    values[op.sum - 'a'] = adder + addend;
    if (values[op.sum - 'a'] != chain[i + 1]) {
      printf("%lu != %c (%lu) = %c (%lu) + %c (%lu)\n", chain[i + 1], op.sum, values[op.sum - 'a'], op.adder, adder, op.addend, addend);
      return false;
    }
  }
  return true;
}

int main() {
  vector<unsigned long> chain = {1, 2, 4, 6, 12, 18, 19, 25, 50, 100, 119, 144, 288, 576, 720, 839, 1678, 2517, 2661, 5322, 6161, 8822, 14983, 23805, 47610, 62593, 125186, 250372, 274177, 548354, 1096708, 1370885, 2467593, 4935186, 9870372, 19740744, 29611116, 49351860, 98703720, 128314836, 256629672, 513259344, 1026518688, 2053037376, 4106074752, 8212149504, 16424299008, 32848598016, 65697196032, 131394392064, 262788784128, 525577568256, 525626920116, 525628291001, 1051256582002, 2102513164004, 4205026328008, 8410052656016, 16820105312032, 33640210624064, 67280421248128, 67280421310721};
  vector<operation> ops = generateChain(chain);
  if (!verifyChain(chain, ops)) {
    printf("Invalid chain ending in %lu\n", chain.back());
  }
  chain = {1, 2, 4, 6, 12, 18, 19, 25, 44, 88, 176, 352, 528, 616, 641, 1282, 2564, 5128, 7692, 12820, 25640, 51280, 102560, 205120, 410240, 417932, 835864, 1671728, 3343456, 6686912, 6699732, 6700373, 6700417};
  ops = generateChain(chain);
  if (!verifyChain(chain, ops)) {
    printf("Invalid chain ending in %lu\n", chain.back());
  }
  chain = {1, 2, 3, 6, 12, 15, 30, 60, 120, 240, 255, 510, 1020, 2040, 4080, 8160, 16320, 32640, 65280, 65535, 131070, 262140, 524280, 1048560, 2097120, 4194240, 8388480, 16776960, 33553920, 67107840, 134215680, 268431360, 536862720, 1073725440, 2147450880, 4294901760, 4294967295, 4294967297};
  ops = generateChain(chain);
  if (!verifyChain(chain, ops)) {
    printf("Invalid chain ending in %lu\n", chain.back());
  }
}