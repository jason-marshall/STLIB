// -*- C++ -*-

#include "stlib/numerical/integer/compare.h"

#include <cassert>

using namespace stlib;

int
main()
{
  using numerical::isNonNegative;

  assert(isNonNegative(int(0)));
  assert(! isNonNegative(int(-1)));
  assert(isNonNegative(unsigned(0)));

  return 0;
}
