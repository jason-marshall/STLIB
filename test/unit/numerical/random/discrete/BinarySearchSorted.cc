// -*- C++ -*-

#include "stlib/numerical/random/discrete/DiscreteGeneratorBinarySearchSorted.h"

#include "stlib/ads/algorithm/statistics.h"

#include "updateSum.ipp"

#include <iostream>
#include <vector>

using namespace stlib;

int
main()
{
  typedef numerical::DiscreteGeneratorBinarySearchSorted<> Generator;

#define USE_INFLUENCE
#include "static.ipp"
#include "dynamic.ipp"

  return 0;
}
