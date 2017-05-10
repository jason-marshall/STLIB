// -*- C++ -*-

#include "stlib/numerical/random/discrete/DiscreteGenerator2DSearchBubbleSort.h"

#include "stlib/ads/algorithm/statistics.h"

#include "updateSum.ipp"

#include <iostream>
#include <vector>

using namespace stlib;

int
main()
{
  typedef numerical::DiscreteGenerator2DSearchBubbleSort<> Generator;

#include "static.ipp"
#include "dynamic.ipp"

  return 0;
}
