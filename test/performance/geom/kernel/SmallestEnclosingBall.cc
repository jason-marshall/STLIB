// -*- C++ -*-

#include "stlib/geom/kernel/SmallestEnclosingBall.h"
#include "stlib/ads/timer.h"

#include <iostream>
#include <vector>

#include <cassert>

using namespace stlib;

int
main()
{
  const std::size_t Dimension = 3;
  const std::size_t NumPoints = 20;
  const std::size_t NumTests = 100000;
  typedef geom::SmallestEnclosingBall<Dimension>::Point Point;

  // Random points.
  std::vector<Point> points(NumPoints * NumTests);
  for (std::size_t i = 0; i != points.size(); ++i) {
    for (std::size_t j = 0; j != Dimension; ++j) {
      points[i][j] = rand();
    }
  }

  geom::SmallestEnclosingBall<Dimension> seb;

  double result = 0;
  std::vector<Point>::const_iterator p = points.begin();

  ads::Timer timer;
  timer.tic();
  for (std::size_t i = 0; i != NumTests; ++i, p += NumPoints) {
    seb.build(p, p + NumPoints);
    result += seb.squaredRadius();
  }
  const double elapsedTime = timer.toc();

  std::cout << "Meaningless result = " << result << '\n'
            << "Average time to construct a ball with " << NumPoints
            << " points = " << elapsedTime / NumTests * 1e9
            << " nanoseconds.\n";

  return 0;
}



