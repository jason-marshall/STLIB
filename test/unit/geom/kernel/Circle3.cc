// -*- C++ -*-

#include "stlib/geom/kernel/Circle3.h"

#include <iostream>

#include <cassert>

USING_STLIB_EXT_ARRAY;
using namespace stlib;

int
main()
{
  typedef geom::Circle3<double> Circle;
  typedef Circle::Point Point;
  {
    // Default constructor
    Circle x;
    x.radius = 0;
    assert(x.radius == 0);
  }
  {
    // Point constructor
    const Point c = {{1, 2, 3}};
    const Point n = {{1, 0, 0}};
    const double r = 1;
    const Circle x = {c, n, r};
    assert(x.isValid());
    std::cout << "Circle((1,2,3),(1,0,0),1) = " << x << "\n";

    // copy constructor
    const Circle y(x);
    assert(y == x);

    // assignment operator
    const Circle z = x;
    assert(z == x);

    // Accessors.
    assert(x.center == c);
    assert(x.normal == n);
    assert(x.radius == r);
  }
  // += operator
  {
    const Point c = {{1, 2, 3}};
    const Point n = {{1, 0, 0}};
    const Point p = {{2, 3, 5}};
    const double r = 1;
    Circle x = {c, n, r};
    x += p;
    const Circle y = {c + p, n, r};
    assert(x == y);
  }
  // -= operator
  {
    const Point c = {{1, 2, 3}};
    const Point n = {{1, 0, 0}};
    const Point p = {{2, 3, 5}};
    const double r = 1;
    Circle x = {c, n, r};
    x -= p;
    const Circle y = {c - p, n, r};
    assert(x == y);
  }
  // == operator
  {
    Circle a = {Point{{1., 2., 3.}},
                Point{{1., 0., 0.}}, 1
               };
    Circle b = {Point{{2., 3., 5.}}, Point{{1., 0., 0.}}, 1};
    assert(!(a == b));
  }
  {
    Circle a = {Point{{1., 2., 3.}}, Point{{0., 1., 0.}}, 1};
    Circle b = {Point{{1., 2., 3.}}, Point{{1., 0., 0.}}, 1};
    assert(!(a == b));
  }
  {
    Circle a = {Point{{1., 2., 3.}}, Point{{1., 0., 0.}}, 1};
    Circle b = {Point{{1., 2., 3.}}, Point{{1., 0., 0.}}, 2};
    assert(!(a == b));
  }
  {
    Circle a = {Point{{1., 2., 3.}}, Point{{1., 0., 0.}}, 1};
    Circle b = {Point{{1., 2., 3.}}, Point{{1., 0., 0.}}, 1};
    assert(a == b);
  }
  // != operator
  {
    Circle a = {Point{{1., 2., 3.}}, Point{{1., 0., 0.}}, 1};
    Circle b = {Point{{2., 3., 5.}}, Point{{1., 0., 0.}}, 1};
    assert(a != b);
  }
  {
    Circle a = {Point{{1., 2., 3.}}, Point{{0., 1., 0.}}, 1};
    Circle b = {Point{{1., 2., 3.}}, Point{{1., 0., 0.}}, 1};
    assert(a != b);
  }
  {
    Circle a = {Point{{1., 2., 3.}}, Point{{1., 0., 0.}}, 1};
    Circle b = {Point{{1., 2., 3.}}, Point{{1., 0., 0.}}, 2};
    assert(a != b);
  }
  {
    Circle a = {Point{{1., 2., 3.}}, Point{{1., 0., 0.}}, 1};
    Circle b = {Point{{1., 2., 3.}}, Point{{1., 0., 0.}}, 1};
    assert(!(a != b));
  }
  // Closest point.
  {
    const Circle x = {Point{{0., 0., 0.}},
                      Point{{1., 0., 0.}}, 1
                     };
    Point cp;
    const double eps = 10.0 * std::numeric_limits<double>::epsilon();

    geom::computeClosestPoint(x, Point{{0., 1., 0.}}, &cp);
    assert(stlib::ext::euclideanDistance(cp, Point{{0., 1., 0.}}) < eps);

    geom::computeClosestPoint(x, Point{{0., 2., 0.}}, &cp);
    assert(stlib::ext::euclideanDistance(cp, Point{{0., 1., 0.}}) < eps);

    geom::computeClosestPoint(x, Point{{0., 0.1, 0.}}, &cp);
    assert(stlib::ext::euclideanDistance(cp, Point{{0., 1., 0.}}) < eps);

    geom::computeClosestPoint(x, Point{{10., 2., 0.}}, &cp);
    assert(stlib::ext::euclideanDistance(cp, Point{{0., 1., 0.}}) < eps);
  }
  {
    const Circle x = {Point{{1., 2., 3.}},
                      Point{{1., 0., 0.}}, 2
                     };
    Point cp;
    const double eps = 10.0 * std::numeric_limits<double>::epsilon();

    geom::computeClosestPoint(x, Point{{1., 4., 3.}}, &cp);
    assert(stlib::ext::euclideanDistance(cp, Point{{1., 4., 3.}}) < eps);

    geom::computeClosestPoint(x, Point{{1., 6., 3.}}, &cp);
    assert(stlib::ext::euclideanDistance(cp, Point{{1., 4., 3.}}) < eps);

    geom::computeClosestPoint(x, Point{{1., 2.1, 3.}}, &cp);
    assert(stlib::ext::euclideanDistance(cp, Point{{1., 4., 3.}}) < eps);

    geom::computeClosestPoint(x, Point{{11., 6., 3.}}, &cp);
    assert(stlib::ext::euclideanDistance(cp, Point{{1., 4., 3.}}) < eps);
  }
  // Closest point to a line segment.
  {
    const Circle x = {Point{{0., 0., 0.}},
                      Point{{1., 0., 0.}}, 1
                     };
    Point cp;
    const double eps = std::sqrt(std::numeric_limits<double>::epsilon());

    geom::computeClosestPoint(x, Point{{-1., 1., 0.}}, Point{{3.,
                                                             1., 0.}}, &cp);
    assert(stlib::ext::euclideanDistance(cp, Point{{0., 1., 0.}}) < eps);

    geom::computeClosestPoint(x, Point{{-1., 2., 0.}}, Point{{3.,
                              2., 0.}}, &cp);
    assert(stlib::ext::euclideanDistance(cp, Point{{0., 1., 0.}}) < eps);

    geom::computeClosestPoint(x, Point{{-1., 1., 0.}}, Point{{3.,
                              2., 0.}}, &cp);
    assert(stlib::ext::euclideanDistance(cp, Point{{0., 1., 0.}}) < eps);

    geom::computeClosestPoint(x, Point{{-1., 0., 0.}}, Point{{3.,
                              0., 0.}}, &cp);
    std::cout << "cp = " << cp << "\n";

    geom::computeClosestPoint(x, Point{{-1., 1., -1.}}, Point{{3.,
                              2., 1.}}, &cp);
    std::cout << "cp = " << cp << "\n";

    geom::computeClosestPoint(x, Point{{-1., 2., 3.}}, Point{{3.,
                              1., -2.}}, &cp);
    std::cout << "cp = " << cp << "\n";
  }
  return 0;
}
