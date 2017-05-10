// -*- C++ -*-

#include "stlib/geom/kernel/BBoxDistanceSimd.h"
#include "stlib/ads/timer.h"
#include "stlib/simd/shuffle.h"

#include <iostream>
#if (__cplusplus >= 201103L)
#include <random>
#endif

#include <vector>

#include <cassert>

using namespace stlib;

int
main()
{
  const std::size_t Dimension = 3;
  typedef geom::BBox<Float, Dimension> BBox;
  const std::size_t Size = 2048;
  typedef simd::Vector<Float>::Type Vector;
  const std::size_t VectorSize = simd::Vector<Float>::Size;

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

  std::vector<Float, simd::allocator<Float> >
  lowerData(Dimension * boxes.size());
  std::vector<Float, simd::allocator<Float> >
  upperData(Dimension * boxes.size());
  std::vector<Float, simd::allocator<Float> > bounds(boxes.size());

  std::size_t k = 0;
  for (std::size_t i = 0; i != boxes.size(); ++i) {
    for (std::size_t j = 0; j != Dimension; ++j) {
      lowerData[k] = boxes[i].lower[j];
      upperData[k] = boxes[i].upper[j];
      ++k;
    }
  }
  simd::aosToHybridSoa<Dimension>(&lowerData);
  simd::aosToHybridSoa<Dimension>(&upperData);

  const std::size_t PtsPerBox = Dimension;
  std::array<std::vector<Float, simd::allocator<Float> >, PtsPerBox>
  extremePointData;
  for (std::size_t i = 0; i != extremePointData.size(); ++i) {
    extremePointData[i].resize(Dimension * boxes.size());
  }
  std::size_t n = 0;
  // For each box.
  for (std::size_t i = 0; i != boxes.size(); ++i) {
    // For each dimension.
    for (std::size_t j = 0; j != Dimension; ++j) {
      // For each of the extreme points in the box. (For simplicity, we just
      // duplicate the lower corner.)
      for (std::size_t k = 0; k != PtsPerBox; ++k) {
        extremePointData[k][n] = boxes[i].lower[j];
      }
      ++n;
    }
  }
  for (std::size_t i = 0; i != extremePointData.size(); ++i) {
    simd::aosToHybridSoa<Dimension>(&extremePointData[i]);
  }

  Float result = 0;
  ads::Timer timer;

  // Warm up.
  for (std::size_t i = 0; i != boxes.size(); ++i) {
    geom::BBoxDistance<Float, Dimension> bboxDistance(boxes[i]);
    bboxDistance.lowerBound2(lowerData, upperData, &bounds);
  }

  // Time it.
  timer.tic();
  for (std::size_t i = 0; i != boxes.size(); ++i) {
    geom::BBoxDistance<Float, Dimension> bboxDistance(boxes[i]);
    bboxDistance.lowerBound2(lowerData, upperData, &bounds);
  }
  const double lowerBound2Time = timer.toc();

  Vector r = simd::setzero<Float>();
  for (std::size_t i = 0; i != bounds.size(); i += VectorSize) {
    r += simd::load(&bounds[i]);
  }
  result += simd::sum(r);

  // Warm up.
  for (std::size_t i = 0; i != boxes.size(); ++i) {
    geom::BBoxDistance<Float, Dimension> bboxDistance(boxes[i]);
    bboxDistance.upperBound2(extremePointData, &bounds);
  }

  // Time it.
  timer.tic();
  for (std::size_t i = 0; i != boxes.size(); ++i) {
    geom::BBoxDistance<Float, Dimension> bboxDistance(boxes[i]);
    bboxDistance.upperBound2(extremePointData, &bounds);
  }
  const double upperBound2Time = timer.toc();

  r = simd::setzero<Float>();
  for (std::size_t i = 0; i != bounds.size(); i += VectorSize) {
    r += simd::load(&bounds[i]);
  }
  result += simd::sum(r);

  std::cout << "Meaningless result = " << result << '\n'
            << "lowerBound2Time = "
            << lowerBound2Time / (boxes.size() * boxes.size()) * 1e9
            << " nanoseconds.\n"
            << "upperBound2Time = "
            << upperBound2Time / (boxes.size() * boxes.size()) * 1e9
            << " nanoseconds.\n";

  return 0;
}



