// -*- C++ -*-

#include "stlib/numerical/random/poisson/PoissonPdfCached.h"

#include <iostream>

using namespace stlib;

int
main()
{
  numerical::PoissonPdfCached<> f(8);

  for (double mean = 0; mean != 8; ++mean) {
    for (int n = 0; n != 8; ++n) {
      std::cout << f(mean, n) << " ";
    }
    std::cout << "\n";
  }

  return 0;
}
