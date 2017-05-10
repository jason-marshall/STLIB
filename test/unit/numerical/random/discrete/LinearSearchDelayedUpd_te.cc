// -*- C++ -*-

// Note this file is intentionally misspelled. On Windows, running any 
// program with the word "update" in its name will bring up a dialog asking,
// "Do you want to allow the following program from an 
// unknown publisher to make changes to this computer?"

#include "stlib/numerical/random/discrete/DiscreteGeneratorLinearSearchDelayedUpdate.h"

#include "stlib/ads/algorithm/statistics.h"

#include "updateSum.ipp"

#include <iostream>
#include <vector>

using namespace stlib;

int
main()
{
  typedef numerical::DiscreteGeneratorLinearSearchDelayedUpdate<> Generator;

#include "static.ipp"
#include "dynamic.ipp"

  return 0;
}
