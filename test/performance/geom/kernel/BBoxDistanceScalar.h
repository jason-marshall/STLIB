// -*- C++ -*-

#include "stlib/geom/kernel/BBoxDistance.h"
#include "stlib/ads/timer.h"

#include <iostream>
#if (__cplusplus >= 201103L)
#include <random>
#endif
#include <vector>

#include <cassert>

USING_STLIB_EXT_ARRAY_MATH_OPERATORS;
using namespace stlib;

int
main()
{
  const std::size_t Dimension = 3;
  typedef geom::BBox<Float, Dimension> BBox;
  typedef BBox::Point Point;
  const std::size_t Size = 1024;

#if (__cplusplus >= 201103L)
  std::default_random_engine generator;
  std::uniform_real_distribution<Float> distribution(0.0, 1.0);
#else
  const Float RandScaling = 1 / Float(RAND_MAX);
#endif

  // Random bounding boxes.
  std::vector<BBox> boxes(Size);
  Float lower, upper;
  for (std::size_t i = 0; i != boxes.size(); ++i) {
    for (std::size_t j = 0; j != Dimension; ++j) {
#if (__cplusplus >= 201103L)
      lower = distribution(generator);
      upper = distribution(generator);
#else
      lower = rand() * RandScaling;
      upper = rand() * RandScaling;
#endif
      if (lower > upper) {
        std::swap(lower, upper);
      }
      boxes[i].lower[j] = lower;
      boxes[i].upper[j] = upper;
    }
  }

  // Points that lie on a line segment.
  std::vector<std::array<Point, Dimension> > pointsOn(Size);
  for (std::size_t i = 0; i != pointsOn.size(); ++i) {
    for (std::size_t j = 0; j != Dimension; ++j) {
      Float f = j / (Dimension - 1);
      pointsOn[i][j] = f * boxes[i].lower + (1 - f) * boxes[i].upper;
    }
  }

  Float result = 0;
  ads::Timer timer;

  // Warm up.
  for (std::size_t i = 0; i != boxes.size(); ++i) {
    for (std::size_t j = 0; j != boxes.size(); ++j) {
      result += geom::minMinDist2(boxes[i], boxes[j]);
    }
  }

  // Time it.
  timer.tic();
  for (std::size_t i = 0; i != boxes.size(); ++i) {
    for (std::size_t j = 0; j != boxes.size(); ++j) {
      result += geom::minMinDist2(boxes[i], boxes[j]);
    }
  }
  const double minMinTime = timer.toc();

  // Warm up.
  for (std::size_t i = 0; i != boxes.size(); ++i) {
    for (std::size_t j = 0; j != boxes.size(); ++j) {
      result += geom::maxMaxDist2(boxes[i], boxes[j]);
    }
  }

  // Time it.
  timer.tic();
  for (std::size_t i = 0; i != boxes.size(); ++i) {
    for (std::size_t j = 0; j != boxes.size(); ++j) {
      result += geom::maxMaxDist2(boxes[i], boxes[j]);
    }
  }
  const double maxMaxTime = timer.toc();

  // Warm up.
  for (std::size_t i = 0; i != boxes.size(); ++i) {
    for (std::size_t j = 0; j != boxes.size(); ++j) {
      for (std::size_t k = 0; k != Dimension; ++k) {
        result += geom::maxMinDist(boxes[i].lower[k], boxes[i].upper[k],
                                   boxes[j].lower[k], boxes[j].upper[k]);
      }
    }
  }

  // Time it.
  timer.tic();
  for (std::size_t i = 0; i != boxes.size(); ++i) {
    for (std::size_t j = 0; j != boxes.size(); ++j) {
      for (std::size_t k = 0; k != Dimension; ++k) {
        result += geom::maxMinDist(boxes[i].lower[k], boxes[i].upper[k],
                                   boxes[j].lower[k], boxes[j].upper[k]);
      }
    }
  }
  const double maxMinTime = timer.toc();

  // Warm up.
  for (std::size_t i = 0; i != boxes.size(); ++i) {
    for (std::size_t j = 0; j != boxes.size(); ++j) {
      result += geom::nxnDist2(boxes[i], boxes[j]);
    }
  }

  // Time it.
  timer.tic();
  for (std::size_t i = 0; i != boxes.size(); ++i) {
    for (std::size_t j = 0; j != boxes.size(); ++j) {
      result += geom::nxnDist2(boxes[i], boxes[j]);
    }
  }
  const double nxnTime = timer.toc();

  // Warm up.
  for (std::size_t i = 0; i != boxes.size(); ++i) {
    for (std::size_t j = 0; j != boxes.size(); ++j) {
      result += geom::upperBoundDist2(boxes[i], pointsOn[j]);
    }
  }

  // Time it.
  timer.tic();
  for (std::size_t i = 0; i != boxes.size(); ++i) {
    for (std::size_t j = 0; j != boxes.size(); ++j) {
      result += geom::upperBoundDist2(boxes[i], pointsOn[j]);
    }
  }
  const double upperBoundTime = timer.toc();

  std::cout << "Total distance = " << result << '\n'
            << "minMinDist2 = "
            << minMinTime / (boxes.size() * boxes.size()) * 1e9
            << " nanoseconds.\n"
            << "maxMaxDist2 = "
            << maxMaxTime / (boxes.size() * boxes.size()) * 1e9
            << " nanoseconds.\n"
            << "maxMinDist = "
            << maxMinTime / (boxes.size() * boxes.size()) * 1e9
            << " nanoseconds.\n"
            << "nxnDist2 = "
            << nxnTime / (boxes.size() * boxes.size()) * 1e9
            << " nanoseconds.\n"
            << "upperBoundDist2 = "
            << upperBoundTime / (boxes.size() * pointsOn.size()) * 1e9
            << " nanoseconds.\n";

  return 0;
}



