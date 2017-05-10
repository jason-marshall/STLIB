// -*- C++ -*-

#include "stlib/numerical/random/poisson/PoissonCdf.h"

#include <iostream>

using namespace stlib;

int
main()
{
  numerical::PoissonCdf<> f;

  std::cout << "Test for means in the range [0..8).\n";
  for (double mean = 0; mean != 8; ++mean) {
    for (int n = 0; n != 8; ++n) {
      std::cout << f(mean, n) << " ";
    }
    std::cout << "\n";
  }

  std::cout
      << "Test for large means.\n"
      << "PoissonCdf(1000, 0) = " << f(1000, 0) << "\n"
      << "PoissonCdf(1000, 1000) = " << f(1000, 1000) << "\n"
      << "PoissonCdf(1000, 2000) = " << f(1000, 2000) << "\n"
      << "PoissonCdf(1000, 10000) = " << f(1000, 10000) << "\n"
      << "PoissonCdf(1000000, 0) = " << f(1000000, 0) << "\n"
      << "PoissonCdf(1000000, 1000000) = " << f(1000000, 1000000) << "\n"
      << "PoissonCdf(1000000, 2000000) = " << f(1000000, 2000000) << "\n"
      << "PoissonCdf(1000000, 10000000) = " << f(1000000, 10000000) << "\n";

  return 0;
}
