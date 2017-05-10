// -*- C++ -*-

#include "stlib/numerical/random/discrete/DiscreteGenerator2DSearchSorted.h"

#include "stlib/ads/algorithm/statistics.h"

#include "updateSum.ipp"

#include <iostream>
#include <vector>

using namespace stlib;

int
main()
{
  typedef numerical::DiscreteGenerator2DSearchSorted<> Generator;

#include "static.ipp"
#include "dynamic.ipp"

  return 0;
}
