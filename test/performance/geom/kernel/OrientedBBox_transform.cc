// -*- C++ -*-

#include "stlib/geom/kernel/OrientedBBox.h"
#include "stlib/ads/timer.h"
#include "stlib/simd/shuffle.h"

#include <iostream>
#include <vector>


using namespace stlib;

int
main()
{
  USING_STLIB_EXT_ARRAY_MATH_OPERATORS;
  USING_STLIB_EXT_ARRAY_IO_OPERATORS;

  typedef float Float;
  std::size_t const Dimension = 3;
  std::size_t const NumPoints = 24;
  std::size_t const NumTests = 100000;
  typedef geom::OrientedBBox<Float, Dimension> OrientedBBox;
  typedef OrientedBBox::Point Point;

  // Random points.
  std::vector<Point> points(NumPoints);
  for (std::size_t i = 0; i != points.size(); ++i) {
    for (std::size_t j = 0; j != Dimension; ++j) {
      points[i][j] = (Float(1) / RAND_MAX) * rand();
    }
  }

  OrientedBBox obb;
  obb.buildPca(points);
  ads::Timer timer;

  {
    Point result = ext::filled_array<Point>(0);

    timer.tic();
    for (std::size_t i = 0; i != NumTests; ++i) {
      for (std::size_t j = 0; j != points.size(); ++j) {
        result += obb.transform(points[j]);
      }
    }
    double const elapsedTime = timer.toc();

    std::cout << "Scalar version:\n"
              << "Meaningless result = " << result << '\n'
              << "Average time to transform a point = "
              << elapsedTime / NumTests / points.size() * 1e9
              << " nanoseconds.\n";
  }

  {
    Float result = 0;
    std::vector<Float, simd::allocator<Float> > pointData;
    simd::aosToHybridSoa(points, &pointData);
    std::vector<Float, simd::allocator<Float> > transformed;

    timer.tic();
    for (std::size_t i = 0; i != NumTests; ++i) {
      obb.transform(pointData, &transformed);
      for (std::size_t j = 0; j != transformed.size(); ++j) {
        result += transformed[j];
      }
    }
    double const elapsedTime = timer.toc();

    std::cout << "SIMD version:\n"
              << "Meaningless result = " << result << '\n'
              << "Average time to transform a point = "
              << elapsedTime / NumTests / points.size() * 1e9
              << " nanoseconds.\n";
  }

  return 0;
}



