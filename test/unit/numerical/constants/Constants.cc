// -*- C++ -*-

#include "stlib/numerical/constants.h"
#include "stlib/numerical/equality.h"

#include <limits>
#include <cassert>

#include <cmath>

using namespace stlib;
using numerical::areEqual;
using numerical::isSmall;

template<typename Number>
inline
void
test()
{
  Number x = 3.14159265358979323846;
  assert(areEqual(numerical::Constants<Number>::Pi(), x));
  x = 2.718281828459045235360287471352;
  assert(areEqual(numerical::Constants<Number>::E(), x));
  x = numerical::Constants<Number>::Pi() / 180.;
  assert(areEqual(numerical::Constants<Number>::Degree(), x));
  x = numerical::Constants<Number>::Pi() / 180.;
  assert(areEqual(numerical::Constants<Number>::radians(1), x));
  x = 180;
  assert(areEqual(numerical::Constants<Number>::
                  degrees(numerical::Constants<Number>::Pi()), x));
  assert(isSmall(std::sin(numerical::Constants<Number>::Pi())));
  x = -1;
  assert(areEqual(std::cos(numerical::Constants<Number>::Pi()), x));
  x = 1;
  assert(areEqual(std::log(numerical::Constants<Number>::E()), x));
}

int
main()
{
  test<double>();
  test<float>();

  return 0;
}
