// -*- C++ -*-

#include "stlib/ads/functor/composite_compare.h"

#include "stlib/ads/array/FixedArray.h"

#include <cassert>

using namespace stlib;

int
main()
{
  using namespace ads;
  typedef FixedArray<3> Point;

  //
  // N-D
  //
  {
    // less, 0 component
    assert(less_composite_fcn<3>(0, Point(1, 2, 3), Point(2, 2, 3)));
    assert(less_composite_fcn<3>(0, Point(1, 2, 3), Point(1, 3, 3)));
    assert(less_composite_fcn<3>(0, Point(1, 2, 3), Point(1, 2, 4)));
    assert(! less_composite_fcn<3>(0, Point(1, 2, 3), Point(1, 2, 3)));
    assert(! less_composite_fcn<3>(0, Point(2, 2, 3), Point(1, 2, 3)));
    assert(! less_composite_fcn<3>(0, Point(1, 3, 3), Point(1, 2, 3)));
    assert(! less_composite_fcn<3>(0, Point(1, 2, 4), Point(1, 2, 3)));
  }
  {
    // less, 0 component, functor
    less_composite<3, Point> comp;
    comp.set(0);
    assert(comp(Point(1, 2, 3), Point(2, 2, 3)));
    assert(comp(Point(1, 2, 3), Point(1, 3, 3)));
    assert(comp(Point(1, 2, 3), Point(1, 2, 4)));
    assert(! comp(Point(1, 2, 3), Point(1, 2, 3)));
    assert(! comp(Point(2, 2, 3), Point(1, 2, 3)));
    assert(! comp(Point(1, 3, 3), Point(1, 2, 3)));
    assert(! comp(Point(1, 2, 4), Point(1, 2, 3)));
  }
  // CONTINUE
#if 0
  {
    // less, 0 component, functor
    less_composite<Point*> comp;
    comp.set(0);
    Point x(1, 2, 3), x0(2, 2, 3), x1(1, 3, 3), x2(1, 2, 4);
    assert(comp(&x, &x0));
    assert(comp(&x, &x1));
    assert(comp(&x, &x2));
    assert(! comp(&x, &x));
    assert(! comp(&x0, &x));
    assert(! comp(&x1, &x));
    assert(! comp(&x2, &x));
  }
#endif
  {
    // yless
    assert(less_composite_fcn<3>(1, Point(1, 2, 3), Point(1, 3, 3)));
    assert(less_composite_fcn<3>(1, Point(1, 2, 3), Point(1, 2, 4)));
    assert(less_composite_fcn<3>(1, Point(1, 2, 3), Point(2, 2, 3)));
    assert(! less_composite_fcn<3>(1, Point(1, 2, 3), Point(1, 2, 3)));
    assert(! less_composite_fcn<3>(1, Point(1, 3, 3), Point(1, 2, 3)));
    assert(! less_composite_fcn<3>(1, Point(1, 2, 4), Point(1, 2, 3)));
    assert(! less_composite_fcn<3>(1, Point(2, 2, 3), Point(1, 2, 3)));
  } {
    // zless
    assert(less_composite_fcn<3>(2, Point(1, 2, 3), Point(1, 2, 4)));
    assert(less_composite_fcn<3>(2, Point(1, 2, 3), Point(2, 2, 3)));
    assert(less_composite_fcn<3>(2, Point(1, 2, 3), Point(1, 3, 3)));
    assert(! less_composite_fcn<3>(2, Point(1, 2, 3), Point(1, 2, 3)));
    assert(! less_composite_fcn<3>(2, Point(1, 2, 4), Point(1, 2, 3)));
    assert(! less_composite_fcn<3>(2, Point(2, 2, 3), Point(1, 2, 3)));
    assert(! less_composite_fcn<3>(2, Point(1, 3, 3), Point(1, 2, 3)));
  }

  //
  // 3-D
  //
  {
    // xless
    assert(xless_composite_compare(Point(1, 2, 3), Point(2, 2, 3)));
    assert(xless_composite_compare(Point(1, 2, 3), Point(1, 3, 3)));
    assert(xless_composite_compare(Point(1, 2, 3), Point(1, 2, 4)));
    assert(! xless_composite_compare(Point(1, 2, 3), Point(1, 2, 3)));
    assert(! xless_composite_compare(Point(2, 2, 3), Point(1, 2, 3)));
    assert(! xless_composite_compare(Point(1, 3, 3), Point(1, 2, 3)));
    assert(! xless_composite_compare(Point(1, 2, 4), Point(1, 2, 3)));
  } {
    // xless functor
    xless_composite<Point> comp;
    assert(comp(Point(1, 2, 3), Point(2, 2, 3)));
    assert(comp(Point(1, 2, 3), Point(1, 3, 3)));
    assert(comp(Point(1, 2, 3), Point(1, 2, 4)));
    assert(! comp(Point(1, 2, 3), Point(1, 2, 3)));
    assert(! comp(Point(2, 2, 3), Point(1, 2, 3)));
    assert(! comp(Point(1, 3, 3), Point(1, 2, 3)));
    assert(! comp(Point(1, 2, 4), Point(1, 2, 3)));
  }
  // CONTINUE
#if 0
  {
    // xless functor
    xless_composite<Point*> comp;
    Point x(1, 2, 3), x0(2, 2, 3), x1(1, 3, 3), x2(1, 2, 4);
    assert(comp(&x, &x0));
    assert(comp(&x, &x1));
    assert(comp(&x, &x2));
    assert(! comp(&x, &x));
    assert(! comp(&x0, &x));
    assert(! comp(&x1, &x));
    assert(! comp(&x2, &x));
  }
#endif
  {
    // yless
    assert(yless_composite_compare(Point(1, 2, 3), Point(1, 3, 3)));
    assert(yless_composite_compare(Point(1, 2, 3), Point(1, 2, 4)));
    assert(yless_composite_compare(Point(1, 2, 3), Point(2, 2, 3)));
    assert(! yless_composite_compare(Point(1, 2, 3), Point(1, 2, 3)));
    assert(! yless_composite_compare(Point(1, 3, 3), Point(1, 2, 3)));
    assert(! yless_composite_compare(Point(1, 2, 4), Point(1, 2, 3)));
    assert(! yless_composite_compare(Point(2, 2, 3), Point(1, 2, 3)));
  } {
    // zless
    assert(zless_composite_compare(Point(1, 2, 3), Point(1, 2, 4)));
    assert(zless_composite_compare(Point(1, 2, 3), Point(2, 2, 3)));
    assert(zless_composite_compare(Point(1, 2, 3), Point(1, 3, 3)));
    assert(! zless_composite_compare(Point(1, 2, 3), Point(1, 2, 3)));
    assert(! zless_composite_compare(Point(1, 2, 4), Point(1, 2, 3)));
    assert(! zless_composite_compare(Point(2, 2, 3), Point(1, 2, 3)));
    assert(! zless_composite_compare(Point(1, 3, 3), Point(1, 2, 3)));
  }

  return 0;
}
