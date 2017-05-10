// -*- C++ -*-

#include "stlib/ads/algorithm/countGroups.h"

#include <array>

#include <cassert>

using namespace stlib;

int
main()
{
  using ads::countGroups;
  {
    std::array<int, 0> a;
    assert(countGroups(a.begin(), a.end()) == 0);
  }
  {
    std::array<int, 1> a = {{0}};
    assert(countGroups(a.begin(), a.end()) == 1);
  }
  {
    std::array<int, 2> a = {{0, 0}};
    assert(countGroups(a.begin(), a.end()) == 1);
  }
  {
    std::array<int, 2> a = {{0, 1}};
    assert(countGroups(a.begin(), a.end()) == 2);
  }
  {
    std::array<int, 3> a = {{0, 0, 0}};
    assert(countGroups(a.begin(), a.end()) == 1);
  }
  {
    std::array<int, 3> a = {{0, 0, 1}};
    assert(countGroups(a.begin(), a.end()) == 2);
  }
  {
    std::array<int, 3> a = {{0, 1, 1}};
    assert(countGroups(a.begin(), a.end()) == 2);
  }
  {
    std::array<int, 3> a = {{0, 1, 0}};
    assert(countGroups(a.begin(), a.end()) == 3);
  }

  return 0;
}
