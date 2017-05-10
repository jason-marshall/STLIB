// -*- C++ -*-

#include "stlib/stochastic/TimeEpochOffset.h"

#include <iostream>

#include <cassert>

using namespace stlib;

int
main()
{
  typedef stochastic::TimeEpochOffset TimeEpochOffset;

  TimeEpochOffset t;
  assert(t == 0);
  t.updateEpoch(1e9);
  t += 1e9;
  t.updateEpoch(1.);
  t += 1.;
  assert(t == 1e9 + 1.);

  // Copy.
  {
    TimeEpochOffset x(t);
    assert(x == t);
  }
  // Assignment.
  {
    TimeEpochOffset x;
    x = t;
    assert(x == t);
  }

  t = 2.;
  assert(t == 2.);

  return 0;
}
