// -*- C++ -*-

#include "stlib/numerical/round/floor.h"

#include <cassert>

using namespace stlib;

int
main()
{
  assert(numerical::floorNonNegative<char>(5.5) == 5);
  assert(numerical::floorNonNegative<unsigned char>(5.5) == 5);
  assert(numerical::floorNonNegative<short>(5.5) == 5);
  assert(numerical::floorNonNegative<unsigned short>(5.5) == 5);
  assert(numerical::floorNonNegative<int>(5.5) == 5);
  assert(numerical::floorNonNegative<unsigned int>(5.5) == 5);
  assert(numerical::floorNonNegative<long>(5.5) == 5);
  assert(numerical::floorNonNegative<unsigned long>(5.5) == 5);
  assert(numerical::floorNonNegative<double>(5.5) == 5);
  assert(numerical::floorNonNegative<float>(float(5.5)) == 5);
  assert(numerical::floorNonNegative<double>(1e20) == 1e20);

  assert(numerical::floor<char>(-4.5) == -5);
  assert(numerical::floor<char>(-5.) == -5);
  assert(numerical::floor<char>(4.5) == 4);
  assert(numerical::floor<char>(5.) == 5);
  assert(numerical::floor<short>(-4.5) == -5);
  assert(numerical::floor<short>(-5.) == -5);
  assert(numerical::floor<short>(4.5) == 4);
  assert(numerical::floor<short>(5.) == 5);
  assert(numerical::floor<int>(-4.5) == -5);
  assert(numerical::floor<int>(-5.) == -5);
  assert(numerical::floor<int>(4.5) == 4);
  assert(numerical::floor<int>(5.) == 5);
  assert(numerical::floor<long>(-4.5) == -5);
  assert(numerical::floor<long>(-5.) == -5);
  assert(numerical::floor<long>(4.5) == 4);
  assert(numerical::floor<long>(5.) == 5);
  assert(numerical::floor<double>(-4.5) == -5);
  assert(numerical::floor<double>(-5.) == -5);
  assert(numerical::floor<double>(4.5) == 4);
  assert(numerical::floor<double>(5.) == 5);
  assert(numerical::floor<float>(float(-4.5)) == -5);
  assert(numerical::floor<float>(float(-5.)) == -5);
  assert(numerical::floor<float>(float(4.5)) == 4);
  assert(numerical::floor<float>(float(5.)) == 5);
  assert(numerical::floor<double>(-1e20) == -1e20);
  assert(numerical::floor<double>(1e20) == 1e20);

  return 0;
}
