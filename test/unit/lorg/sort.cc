// -*- C++ -*-

#include "stlib/lorg/sort.h"

#include <cassert>

using namespace stlib;

int
main()
{
  typedef std::pair<std::size_t, std::size_t> Value;

  // Specify the number of digits.
  for (int digits = 0; digits != 10; ++digits) {
    std::vector<Value> pairs(1 << digits);
    for (std::size_t i = 0; i != pairs.size(); ++i) {
      pairs[i].first = pairs[i].second = pairs.size() - i - 1;
    }
    {
      lorg::RciSort<std::size_t, std::size_t> rci(&pairs, digits);
      rci.sort();
    }
    for (std::size_t i = 0; i != pairs.size(); ++i) {
      assert(pairs[i].first == i);
      assert(pairs[i].second == pairs[i].first);
    }
  }

  // Don't specify the number of digits.
  {
    std::vector<Value> pairs(1024);
    for (std::size_t i = 0; i != pairs.size(); ++i) {
      pairs[i].first = pairs[i].second = i;
    }
    {
      lorg::RciSort<std::size_t, std::size_t> rci(&pairs);
      rci.sort();
    }
    for (std::size_t i = 0; i != pairs.size(); ++i) {
      assert(pairs[i].first == i);
      assert(pairs[i].second == pairs[i].first);
    }

    for (std::size_t i = 0; i != pairs.size(); ++i) {
      pairs[i].first = pairs[i].second = pairs.size() - i - 1;
    }
    {
      lorg::RciSort<std::size_t, std::size_t> rci(&pairs);
      rci.sort();
    }
    for (std::size_t i = 0; i != pairs.size(); ++i) {
      assert(pairs[i].first == i);
      assert(pairs[i].second == pairs[i].first);
    }
  }

  return 0;
}
