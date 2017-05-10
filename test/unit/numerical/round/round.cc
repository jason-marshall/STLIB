// -*- C++ -*-

#include "stlib/numerical/round/round.h"

#include <cassert>

using namespace stlib;

int
main()
{
  assert(numerical::roundNonNegative<char>(5.4) == 5);
  assert(numerical::roundNonNegative<unsigned char>(5.4) == 5);
  assert(numerical::roundNonNegative<short>(5.4) == 5);
  assert(numerical::roundNonNegative<unsigned short>(5.4) == 5);
  assert(numerical::roundNonNegative<int>(5.4) == 5);
  assert(numerical::roundNonNegative<unsigned int>(5.4) == 5);
  assert(numerical::roundNonNegative<long>(5.4) == 5);
  assert(numerical::roundNonNegative<unsigned long>(5.4) == 5);
  assert(numerical::roundNonNegative<double>(5.4) == 5);
  assert(numerical::roundNonNegative<float>(float(5.4)) == 5);

  assert(numerical::roundNonNegative<char>(5.6) == 6);
  assert(numerical::roundNonNegative<unsigned char>(5.6) == 6);
  assert(numerical::roundNonNegative<short>(5.6) == 6);
  assert(numerical::roundNonNegative<unsigned short>(5.6) == 6);
  assert(numerical::roundNonNegative<int>(5.6) == 6);
  assert(numerical::roundNonNegative<unsigned int>(5.6) == 6);
  assert(numerical::roundNonNegative<long>(5.6) == 6);
  assert(numerical::roundNonNegative<unsigned long>(5.6) == 6);
  assert(numerical::roundNonNegative<double>(5.6) == 6);
  assert(numerical::roundNonNegative<float>(float(5.6)) == 6);

  assert(numerical::roundNonNegative<double>(1e20) == 1e20);


  assert(numerical::round<char>(-4.4) == -4);
  assert(numerical::round<char>(-4.6) == -5);
  assert(numerical::round<char>(4.4) == 4);
  assert(numerical::round<char>(4.6) == 5);

  assert(numerical::round<short>(-4.4) == -4);
  assert(numerical::round<short>(-4.6) == -5);
  assert(numerical::round<short>(4.4) == 4);
  assert(numerical::round<short>(4.6) == 5);

  assert(numerical::round<int>(-4.4) == -4);
  assert(numerical::round<int>(-4.6) == -5);
  assert(numerical::round<int>(4.4) == 4);
  assert(numerical::round<int>(4.6) == 5);

  assert(numerical::round<long>(-4.4) == -4);
  assert(numerical::round<long>(-4.6) == -5);
  assert(numerical::round<long>(4.4) == 4);
  assert(numerical::round<long>(4.6) == 5);

  assert(numerical::round<double>(-4.4) == -4);
  assert(numerical::round<double>(-4.6) == -5);
  assert(numerical::round<double>(4.4) == 4);
  assert(numerical::round<double>(4.6) == 5);

  assert(numerical::round<float>(float(-4.4)) == -4);
  assert(numerical::round<float>(float(-4.6)) == -5);
  assert(numerical::round<float>(float(4.4)) == 4);
  assert(numerical::round<float>(float(4.6)) == 5);

  assert(numerical::round<double>(-1e20) == -1e20);
  assert(numerical::round<double>(1e20) == 1e20);

  return 0;
}

