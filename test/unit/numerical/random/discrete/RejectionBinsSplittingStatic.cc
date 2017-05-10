// -*- C++ -*-

#include "stlib/numerical/random/discrete/DiscreteGeneratorRejectionBinsSplittingStatic.h"

#include "stlib/ads/algorithm/statistics.h"

#include "updateSum.ipp"

#include <iostream>
#include <vector>


using namespace stlib;

int
main()
{
  {
    typedef numerical::DiscreteGeneratorRejectionBinsSplittingStatic<false>
    Generator;
#include "static.ipp"
  }
  {
    typedef numerical::DiscreteGeneratorRejectionBinsSplittingStatic<>
    Generator;
#include "static.ipp"
  }

  return 0;
}
