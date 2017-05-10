// -*- C++ -*-

#include "stlib/numerical/random/discrete/DiscreteGeneratorRejectionBinsSplitting.h"

#include "stlib/ads/algorithm/statistics.h"

#include "updateSum.ipp"

#include <iostream>
#include <vector>


using namespace stlib;

int
main()
{
  {
    typedef numerical::DiscreteGeneratorRejectionBinsSplitting<false> Generator;
#include "static.ipp"
#include "dynamic.ipp"
  }
  {
    typedef numerical::DiscreteGeneratorRejectionBinsSplitting<> Generator;
#include "static.ipp"
#include "dynamic.ipp"
  }

  return 0;
}
