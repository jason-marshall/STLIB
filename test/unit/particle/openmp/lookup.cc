// -*- C++ -*-

#include "stlib/particle/lookup.h"

#include <limits>
#include <cassert>

using namespace stlib;

typedef particle::IntegerTypes::Code Code;

void
test(const particle::LookupTable& table,
     const std::vector<Code>& codes)
{
  assert(codes.back() == std::numeric_limits<Code>::max());
  if (table.shift() == 0) {
    for (std::size_t i = 0; i != codes.size(); ++i) {
      assert(table(codes[i]) == i);
    }
  }
  else {
    for (std::size_t i = 0; i != codes.size(); ++i) {
      assert(table(codes[i]) <= i);
    }
  }
  assert(table(0) == 0);
  assert(table(std::numeric_limits<Code>::max()) == codes.size() - 1);
}

int
main()
{
  // 0 codes.
  {
    std::vector<Code> codes(1);
    codes.back() = std::numeric_limits<Code>::max();

    particle::LookupTable table(codes, codes.size() + 1);
    assert(table.shift() == 0);
    assert(table(0) == 0);
  }
  // 10 codes.
  {
    std::vector<Code> codes(10 + 1);
    for (std::size_t i = 0; i != codes.size() - 1; ++i) {
      codes[i] = i;
    }
    codes.back() = std::numeric_limits<Code>::max();

    particle::LookupTable table(codes, codes.size());
    assert(table.shift() == 0);
    test(table, codes);

    table.initialize(codes, codes.size() - 1);
    assert(table.shift() == 1);
    test(table, codes);

    table.initialize(codes, 2);
    assert(table.shift() == 4);
    test(table, codes);

    for (std::size_t i = 0; i != codes.size() - 1; ++i) {
      codes[i] = 2 * i + 3;
    }

    table.initialize(codes, codes.size());
    assert(table.shift() == 1);
    test(table, codes);

    table.initialize(codes, codes.size() - 1);
    assert(table.shift() == 2);
    test(table, codes);

    table.initialize(codes, 2);
    assert(table.shift() == 5);
    test(table, codes);
  }

  return 0;
}
