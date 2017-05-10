// -*- C++ -*-

#include "stlib/numerical/random/discrete/DiscreteGeneratorLinearSearchSortedStatic.h"

#include "stlib/ads/algorithm/statistics.h"

#include "updateSum.ipp"

#include <iostream>
#include <vector>

using namespace stlib;

int
main()
{
  typedef numerical::DiscreteGeneratorLinearSearchSortedStatic<> Generator;

#include "static.ipp"

  return 0;
}
