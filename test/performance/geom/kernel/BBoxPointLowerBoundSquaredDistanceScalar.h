// -*- C++ -*-

#include "stlib/geom/kernel/distance.h"
#include "stlib/geom/kernel/BBoxPointLowerBoundSquaredDistance.h"
#include "stlib/performance/SimpleTimer.h"

#include <iostream>
#include <random>
#include <vector>

#include <cassert>

int
main()
{
  std::size_t const Dimension = 3;
  typedef stlib::geom::BBox<Float, Dimension> BBox;
  typedef BBox::Point Point;
  std::size_t const Size = 1 << 16;

  std::default_random_engine generator;
  std::uniform_real_distribution<Float> distribution(0.0, 1.0);

  // Random bounding boxes.
  std::vector<BBox> boxes(Size);
  Float lower, upper;
  for (std::size_t i = 0; i != boxes.size(); ++i) {
    for (std::size_t j = 0; j != Dimension; ++j) {
      lower = distribution(generator);
      upper = distribution(generator);
      if (lower > upper) {
        std::swap(lower, upper);
      }
      boxes[i].lower[j] = lower;
      boxes[i].upper[j] = upper;
    }
  }
  Point const queryPoint = stlib::ext::filled_array<Point>(0.5);

  Float result = 0;
  stlib::performance::SimpleTimer timer;

  // Warm up.
  for (std::size_t i = 0; i != boxes.size(); ++i) {
    result += stlib::geom::computeLowerBoundSquaredDistance(boxes[i],
                                                            queryPoint);
  }

  // Time it.
  timer.start();
  for (std::size_t i = 0; i != boxes.size(); ++i) {
    result += stlib::geom::computeLowerBoundSquaredDistance(boxes[i],
                                                            queryPoint);
  }
  timer.stop();

  std::cout << "Total distance = " << result << '\n'
            << "Time per distance calculation = "
            << timer.elapsed() / boxes.size() * 1e9
            << " nanoseconds.\n";


  typedef stlib::geom::BBoxPointLowerBoundSquaredDistance<Float, Dimension>
    BBoxSimd;
  typedef typename BBoxSimd::AlignedPoint AlignedPoint;

  std::vector<BBoxSimd> boxesSimd(boxes.size());
  for (std::size_t i = 0; i != boxesSimd.size(); ++i) {
    boxesSimd[i] = BBoxSimd{boxes[i]};
  }

  AlignedPoint const queryPointSimd{{0.5, 0.5, 0.5}};

  result = 0;

  // Warm up.
  for (std::size_t i = 0; i != boxes.size(); ++i) {
    result += boxesSimd[i](queryPointSimd);
  }

  // Time it.
  timer.start();
  for (std::size_t i = 0; i != boxes.size(); ++i) {
    result += boxesSimd[i](queryPointSimd);
  }
  timer.stop();

  std::cout << "Total distance = " << result << '\n'
            << "Time per distance calculation = "
            << timer.elapsed() / boxesSimd.size() * 1e9
            << " nanoseconds.\n";

  return 0;
}



