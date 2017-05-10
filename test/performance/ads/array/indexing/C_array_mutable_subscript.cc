// -*- C++ -*-

//
// Tests for C_array.
//

#include "stlib/ads/timer.h"

#include <iostream>

using namespace stlib;

int
main()
{
  double p[3];
  p[0] = 0;
  p[1] = 0;
  p[2] = 0;
  const double v = 1e-8;

  ads::Timer timer;
  timer.tic();
  for (int i = 0; i != 100000000; ++i) {
    p[0] += v;
    p[1] += v;
    p[2] += v;
  }
  double t = timer.toc();
  std::cout << "C_array_mutable_subscricpt = " << t << '\n';

  return 0;
}
