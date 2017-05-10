// -*- C++ -*-

#include "stlib/numerical/random/discrete/DiscreteGeneratorLinearSearchSorted.h"

#include "stlib/ads/algorithm/statistics.h"

#include "updateSum.ipp"

#include <iostream>
#include <vector>

using namespace stlib;

int
main()
{
  typedef numerical::DiscreteGeneratorLinearSearchSorted<> Generator;

#include "static.ipp"
#include "dynamic.ipp"

  return 0;
}
