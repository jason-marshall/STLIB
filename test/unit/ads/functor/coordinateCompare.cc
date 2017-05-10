// -*- C++ -*-

#include "stlib/ads/functor/coordinateCompare.h"

#include "stlib/ads/array/FixedArray.h"

#include <cassert>

using namespace stlib;

int
main()
{
  typedef ads::FixedArray<3> Point;
  typedef ads::LessThanCompareCoordinate<Point> Compare;

  {
    Compare compare(0);
    const Point x(1, 2, 3);
    const Point y(2, 3, 5);
    assert(compare(x, y));
    assert(! compare(y, x));
    compare.setCoordinate(1);
    assert(compare(x, y));
    assert(! compare(y, x));
    compare.setCoordinate(2);
    assert(compare(x, y));
    assert(! compare(y, x));
  }
  {
    Compare compare = ads::constructLessThanCompareCoordinate<Point>(0);
    const Point x(1, 2, 3);
    const Point y(2, 3, 5);
    assert(compare(x, y));
  }

  return 0;
}
