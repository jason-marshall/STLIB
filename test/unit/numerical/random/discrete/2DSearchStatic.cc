// -*- C++ -*-

#include "stlib/numerical/random/discrete/DiscreteGenerator2DSearchStatic.h"

#include "stlib/ads/algorithm/statistics.h"

#include "updateSum.ipp"

#include <iostream>
#include <vector>

using namespace stlib;

int
main()
{
  typedef numerical::DiscreteGenerator2DSearchStatic<> Generator;

#include "static.ipp"

  return 0;
}
