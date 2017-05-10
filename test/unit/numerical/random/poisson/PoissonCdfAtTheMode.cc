// -*- C++ -*-

#include "stlib/numerical/random/poisson/PoissonCdfAtTheMode.h"

#include <iostream>

using namespace stlib;

int
main()
{
  numerical::PoissonCdf<> f;
  numerical::PoissonCdfAtTheMode<> g(40);

  double x, a, b;

  std::cout << "At integer values.\n";
  for (int n = 0; n != 40; ++n) {
    x = n;
    a = f(x, n);
    b = g(x);
    std::cout << n << " " << a << " " << b << " " << a - b << "\n";
  }

  std::cout << "At half integer values.\n";
  for (int n = 0; n != 40; ++n) {
    x = n + 0.5;
    a = f(x, n);
    b = g(x);
    std::cout << n << " " << a << " " << b << " " << a - b << "\n";
  }

  return 0;
}
