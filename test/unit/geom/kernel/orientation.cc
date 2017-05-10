// -*- C++ -*-

#include "stlib/geom/kernel/orientation.h"

#include <iostream>
#include <limits>

using namespace stlib;

int
main()
{
  // Orientation determinant.
  {
    typedef std::array<double, 2> Point;
    {
      Point a = {{0, 0}}, b = {{1, 0}}, c = {{1, 1}};
      assert(geom::computeOrientationDeterminant(a, b, c) > 0);
    }
    {
      Point a = {{0, 0}}, b = {{1, 0}}, c = {{2, 0}};
      assert(geom::computeOrientationDeterminant(a, b, c) == 0);
    }
    {
      Point a = {{0, 0}}, b = {{1, 0}}, c = {{1, -1}};
      assert(geom::computeOrientationDeterminant(a, b, c) < 0);
    }
  }


  // In circle.
  {
    typedef std::array<double, 2> Point;
    {
      Point a = {{0, 0}}, b = {{0, 1}}, c = {{1, 0}}, d = {{0.5, 0.5}};
      assert(geom::isInCircle(a, b, c, d));
    }
    {
      Point a = {{0, 0}}, b = {{0, 1}}, c = {{1, 0}}, d = {{0.5, 0.5}};
      assert(geom::isInCircle(b, a, c, d));
    }
    {
      Point a = {{0, 0}}, b = {{0, 1}}, c = {{1, 0}}, d = {{1.5, 1.5}};
      assert(! geom::isInCircle(a, b, c, d));
    }
    {
      Point a = {{0, 0}}, b = {{0, 1}}, c = {{1, 0}}, d = {{1.5, 1.5}};
      assert(! geom::isInCircle(b, a, c, d));
    }
  }

  return 0;
}
