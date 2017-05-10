// -*- C++ -*-

#include "stlib/numerical/specialFunctions/Factorial.h"

#include <iostream>

using namespace stlib;

int
main()
{
  assert(numerical::computeFactorial<int>(0) == 1);
  assert(numerical::computeFactorial<int>(1) == 1);
  assert(numerical::computeFactorial<int>(2) == 2);
  assert(numerical::computeFactorial<int>(3) == 6);
  assert(numerical::computeFactorial<int>(4) == 24);

  {
    numerical::Factorial<> f;
    assert(f(0) == 1);
    assert(f(1) == 1);
    assert(f(2) == 2);
    assert(f(3) == 6);
    assert(f(4) == 24);
    f = numerical::constructFactorial<int>();
  }

  {
    numerical::Factorial<double> f;
    assert(f(0) == 1);
    assert(f(1) == 1);
    assert(f(2) == 2);
    assert(f(3) == 6);
    assert(f(4) == 24);
    f = numerical::constructFactorial<double>();
  }

  return 0;
}
