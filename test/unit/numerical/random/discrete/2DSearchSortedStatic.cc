// -*- C++ -*-

#include "stlib/numerical/random/discrete/DiscreteGenerator2DSearchSortedStatic.h"

#include "stlib/ads/algorithm/statistics.h"

#include "updateSum.ipp"

#include <iostream>
#include <vector>

using namespace stlib;

int
main()
{
  typedef numerical::DiscreteGenerator2DSearchSortedStatic<> Generator;

#include "static.ipp"

  return 0;
}
