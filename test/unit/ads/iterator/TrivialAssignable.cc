// -*- C++ -*-

#include "stlib/ads/iterator/TrivialAssignable.h"

using namespace stlib;

int
main()
{
  ads::TrivialAssignable x;

  x = 1;
  x = 3.45;

  ads::TrivialAssignable y(x);
  x = y;

  return 0;
}
