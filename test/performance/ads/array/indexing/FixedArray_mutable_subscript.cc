// -*- C++ -*-

//
// Timings for FixedArray.
//

#include "stlib/ads/timer.h"

#include "stlib/ads/array/FixedArray.h"
#include <iostream>

using namespace stlib;

int
main()
{
  ads::FixedArray<3> p(0.);
  const double v = 1e-8;

  ads::Timer timer;
  timer.tic();
  for (int i = 0; i != 100000000; ++i) {
    p += v;
  }
  double t = timer.toc();
  std::cout << "FixedArray_mutable_subscricpt = " << t << '\n';

  return 0;
}
