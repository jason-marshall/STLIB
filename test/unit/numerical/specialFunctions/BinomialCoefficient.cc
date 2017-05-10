// -*- C++ -*-

#include "stlib/numerical/specialFunctions/BinomialCoefficient.h"

#include <iostream>

using namespace stlib;

int
main()
{
  assert(numerical::computeBinomialCoefficient<int>(0, 0) == 1);
  assert(numerical::computeBinomialCoefficient<int>(5, 0) == 1);
  assert(numerical::computeBinomialCoefficient<int>(5, 1) == 5);
  assert(numerical::computeBinomialCoefficient<int>(5, 2) == 10);
  assert(numerical::computeBinomialCoefficient<int>(5, 3) == 10);
  assert(numerical::computeBinomialCoefficient<int>(5, 4) == 5);
  assert(numerical::computeBinomialCoefficient<int>(5, 5) == 1);

  {
    numerical::BinomialCoefficient<> f;
    assert(f(0, 0) == 1);
    assert(f(5, 0) == 1);
    assert(f(5, 1) == 5);
    assert(f(5, 2) == 10);
    assert(f(5, 3) == 10);
    assert(f(5, 4) == 5);
    assert(f(5, 5) == 1);
    f = numerical::constructBinomialCoefficient<int>();
  }

  {
    numerical::BinomialCoefficient<double> f;
    assert(f(0, 0) == 1);
    assert(f(5, 0) == 1);
    assert(f(5, 1) == 5);
    assert(f(5, 2) == 10);
    assert(f(5, 3) == 10);
    assert(f(5, 4) == 5);
    assert(f(5, 5) == 1);
    f = numerical::constructBinomialCoefficient<double>();
  }

  return 0;
}
