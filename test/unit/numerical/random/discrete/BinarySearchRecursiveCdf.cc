// -*- C++ -*-

#include "stlib/numerical/random/discrete/DiscreteGeneratorBinarySearchRecursiveCdf.h"

#include "stlib/ads/algorithm/statistics.h"

#include "updateSum.ipp"

#include <iostream>
#include <vector>

using namespace stlib;

int
main()
{
  typedef numerical::DiscreteGeneratorBinarySearchRecursiveCdf<> Generator;

#include "static.ipp"
#include "dynamic.ipp"

  return 0;
}
