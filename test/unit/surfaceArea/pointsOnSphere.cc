// -*- C++ -*-

#include "stlib/surfaceArea/pointsOnSphere.h"
#include "stlib/ext/array.h"
#include "stlib/ext/vector.h"

#include <iostream>

#include <cassert>

USING_STLIB_EXT_VECTOR_IO_OPERATORS;
using namespace stlib;

int
main()
{
  typedef std::array<double, 3> Point;
  std::vector<Point> points;

  surfaceArea::distributePointsOnSphereWithGoldenSectionSpiral<Point>
  (0, std::back_inserter(points));
  assert(points.empty());

  for (std::size_t size = 0; size != 10; ++size) {
    surfaceArea::distributePointsOnSphereWithGoldenSectionSpiral<Point>
    (size, std::back_inserter(points));
    std::cout << size << " points.\n" << points << '\n';
    assert(points.size() == size);
    points.clear();
  }

  return 0;
}
