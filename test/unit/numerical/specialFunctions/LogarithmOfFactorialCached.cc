// -*- C++ -*-

#include "stlib/numerical/specialFunctions/LogarithmOfFactorialCached.h"

#include <iostream>
#include <limits>

using namespace stlib;

int
main()
{
  const int Size = 10;
  numerical::LogarithmOfFactorialCached<> f(Size);
  const double Epsilon = 10 * std::numeric_limits<double>::epsilon();

  int factorial = 1;
  double logarithmOfFactorial;
  for (int n = 0; n != Size; ++n) {
    logarithmOfFactorial = std::log(factorial);
    assert(std::abs(f(n) - logarithmOfFactorial) <=
           logarithmOfFactorial * Epsilon);
    factorial *= n + 1;
  }

  return 0;
}
