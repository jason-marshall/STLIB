// -*- C++ -*-

#include "stlib/numerical/round/ceil.h"

#include <cassert>

using namespace stlib;

int
main()
{
  assert(numerical::ceilNonNegative<char>(5.5) == 6);
  assert(numerical::ceilNonNegative<unsigned char>(5.5) == 6);
  assert(numerical::ceilNonNegative<short>(5.5) == 6);
  assert(numerical::ceilNonNegative<unsigned short>(5.5) == 6);
  assert(numerical::ceilNonNegative<int>(5.5) == 6);
  assert(numerical::ceilNonNegative<unsigned int>(5.5) == 6);
  assert(numerical::ceilNonNegative<long>(5.5) == 6);
  assert(numerical::ceilNonNegative<unsigned long>(5.5) == 6);
  assert(numerical::ceilNonNegative<double>(5.5) == 6);
  assert(numerical::ceilNonNegative<float>(float(5.5)) == 6);
  assert(numerical::ceilNonNegative<double>(1e20) == 1e20);


  assert(numerical::ceil<char>(-4.5) == -4);
  assert(numerical::ceil<char>(-5.) == -5);
  assert(numerical::ceil<char>(4.5) == 5);
  assert(numerical::ceil<char>(5.) == 5);

  assert(numerical::ceil<short>(-4.5) == -4);
  assert(numerical::ceil<short>(-5.) == -5);
  assert(numerical::ceil<short>(4.5) == 5);
  assert(numerical::ceil<short>(5.) == 5);

  assert(numerical::ceil<int>(-4.5) == -4);
  assert(numerical::ceil<int>(-5.) == -5);
  assert(numerical::ceil<int>(4.5) == 5);
  assert(numerical::ceil<int>(5.) == 5);

  assert(numerical::ceil<long>(-4.5) == -4);
  assert(numerical::ceil<long>(-5.) == -5);
  assert(numerical::ceil<long>(4.5) == 5);
  assert(numerical::ceil<long>(5.) == 5);

  assert(numerical::ceil<double>(-4.5) == -4);
  assert(numerical::ceil<double>(-5.) == -5);
  assert(numerical::ceil<double>(4.5) == 5);
  assert(numerical::ceil<double>(5.) == 5);

  assert(numerical::ceil<float>(float(-4.5)) == -4);
  assert(numerical::ceil<float>(float(-5.)) == -5);
  assert(numerical::ceil<float>(float(4.5)) == 5);
  assert(numerical::ceil<float>(float(5.)) == 5);

  assert(numerical::ceil<double>(-1e20) == -1e20);
  assert(numerical::ceil<double>(1e20) == 1e20);

  return 0;
}
