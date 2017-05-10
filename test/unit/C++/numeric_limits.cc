// -*- C++ -*-

#include <limits>

#include <cassert>
#include <cmath>

template<typename _Float>
void
infinity()
{
  _Float const Inf = std::numeric_limits<_Float>::infinity();

#ifndef _MSC_VER
  // Indeterminate form that produces infinity.
  assert(_Float(1) / 0 == Inf);
#endif

  // Arithmetic.
  assert(Inf + Inf == Inf);
  assert(Inf - std::numeric_limits<_Float>::max() == Inf);
  assert(isnan(Inf - Inf));
  assert(Inf * Inf == Inf);
  assert(isnan(Inf * 0));
  assert(Inf / std::numeric_limits<_Float>::max() == Inf);
  assert(isnan(Inf / Inf));

  // Math functions.
  assert(std::sqrt(Inf) == Inf);
  assert(std::log(0) == -Inf);
  assert(std::log(Inf) == Inf);
  assert(std::exp(Inf) == Inf);
  assert(isnan(std::cos(Inf)));
  assert(isnan(std::sin(Inf)));
}

template<typename _Float>
void
max()
{
  _Float const Max = std::numeric_limits<_Float>::max();
  _Float const Inf = std::numeric_limits<_Float>::infinity();

  assert(Max + 1 == Max);
  assert(Max + Max == Inf);
  assert(Max * 2 == Inf);
  assert(Max * Max == Inf);
}

int
main()
{
  infinity<float>();
  infinity<double>();

  max<float>();
  max<double>();

  return 0;
}
