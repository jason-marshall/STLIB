// -*- C++ -*-

#include "stlib/ads/algorithm/unique.h"

#include <cassert>

using namespace stlib;

int
main()
{
  {
    const int size = 4;
    const int data[size] = {1, 2, 3, 4};
    assert(ads::areElementsUnique(data, data + size));
  }
  {
    const int size = 4;
    const int data[size] = {1, 2, 3, 1};
    assert(! ads::areElementsUnique(data, data + size));
  }

  return 0;
}
