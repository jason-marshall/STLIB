// -*- C++ -*-

#include "stlib/geom/kernel/OrientedBBox.h"
#include "stlib/ads/timer.h"
#include "stlib/simd/shuffle.h"

#include <iostream>
#include <vector>


using namespace stlib;

typedef float Float;


Float
plane0(Float /*x*/, Float /*y*/)
{
  return 0;
}


Float
plane(Float const x, Float /*y*/)
{
  return x;
}


Float
quadratic(Float const x, Float const y)
{
  return 0.05 * (x * x + y * y);
}


template<typename _Functor>
void
compareContent(_Functor f)
{
  std::size_t const Dimension = 3;
  std::size_t const NumPoints = 10;
  std::size_t const NumTests = 10;
  typedef geom::BBox<Float, Dimension> BBox;
  typedef geom::OrientedBBox<Float, Dimension> OrientedBBox;
  typedef OrientedBBox::Point Point;

  Float aabbContent = 0;
  Float obbContent = 0;
  Float obbrContent = 0;
  std::vector<Point> points(NumPoints);
  OrientedBBox obb;
  for (std::size_t i = 0; i != NumTests; ++i) {
    // Random points.
    for (std::size_t j = 0; j != points.size(); ++j) {
      for (std::size_t k = 0; k != Dimension - 1; ++k) {
        points[j][k] = (Float(1) / RAND_MAX) * rand();
      }
      points[j][Dimension - 1] = f(points[j][0], points[j][1]);
    }
    aabbContent +=
      content(geom::specificBBox<BBox>(points.begin(), points.end()));
    obb.buildPca(points);
    obbContent += content(obb);
    obb.buildPcaRotate(points);
    obbrContent += content(obb);
  }
  std::cout << "Volume for AABB = " << aabbContent << '\n'
            << "Volume for OBB = " << obbContent << '\n'
            << "Volume for rotated OBB = " << obbrContent << '\n';
}


int
main()
{
  std::cout << "Random points in the plane z = 0.\n";
  compareContent(&plane0);

  std::cout << "Random points in the plane z = x.\n";
  compareContent(&plane);

  std::cout << "Random points in the quadratic z = 0.05*(x^2 + y^2).\n";
  compareContent(&quadratic);

  return 0;
}



