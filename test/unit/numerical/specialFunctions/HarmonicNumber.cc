// -*- C++ -*-

#include "stlib/numerical/specialFunctions/HarmonicNumber.h"

#include <iostream>
#include <limits>

using namespace stlib;

int
main()
{
  const int Size = 5;
  const double H[Size] = {0.0, 1.0, 3.0 / 2.0, 11.0 / 6.0, 25.0 / 12.0};
  const double Epsilon = 10.0 * std::numeric_limits<double>::epsilon();
  for (int n = 0; n != Size; ++n) {
    assert(std::abs(numerical::computeHarmonicNumber<double>(n) - H[n]) <
           Epsilon);
  }
  {
    numerical::HarmonicNumber<> f;
    for (int n = 0; n != Size; ++n) {
      assert(std::abs(f(n) - H[n]) < Epsilon);
    }
    f = numerical::constructHarmonicNumber<double>();
  }

  return 0;
}
