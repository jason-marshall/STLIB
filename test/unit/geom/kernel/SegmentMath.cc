// -*- C++ -*-

#include "stlib/geom/kernel/SegmentMath.h"

#include <iostream>

USING_STLIB_EXT_ARRAY_MATH_OPERATORS;
using namespace stlib;

int
main()
{
  typedef geom::SegmentMath<3> SegmentMath;
  typedef SegmentMath::Point Point;
  {
    // Default constructor
    SegmentMath s;
    std::cout << "SegmentMath() = " << '\n'
              << s << '\n';
  }
  {
    // Point constructor
    Point p = {{1, 2, 3}}, q = {{2, 3, 5}};
    SegmentMath s(p, q);
    std::cout << "SegmentMath((1,2,3),(2,3,5)) = " << '\n'
              << s << '\n';
    assert(s.isValid());

    // copy constructor
    SegmentMath t(s);
    assert(t == s);
    // assignment operator
    SegmentMath u = s;
    assert(u == s);
    // make
    SegmentMath v;
    v.make(p, q);
    // This fails because of floating point arithmetic.
    //    assert(v == s);

    // accessors
    assert(s.getSource() == p);
    assert(s.getTarget() == q);
    Point tangent = {{2 - 1, 3 - 2, 5 - 3}};
    stlib::ext::normalize(&tangent);
    assert(stlib::ext::euclideanDistance(s.getTangent(), tangent) < 1e-8);
    assert(std::abs(s.getLength() - sqrt(6.0)) < 1e-8);
  }
  {
    // += operator
    Point p = {{1, 2, 3}}, q = {{2, 3, 5}};
    SegmentMath s(p, q);
    s += p;
    SegmentMath t(p + p, q + p);
    assert(s == t);
  }
  {
    // -= operator
    Point p = {{1, 2, 3}}, q = {{2, 3, 5}};
    SegmentMath s(p, q);
    s -= p;
    SegmentMath t(p - p, q - p);
    assert(s == t);
  }
  // == operator
  {
    SegmentMath a(Point{{1., 2., 3.}}, Point{{4., 5., 6.}});
    SegmentMath b(Point{{2., 3., 5.}}, Point{{7., 11., 13.}});
    assert(!(a == b));
  }
  {
    SegmentMath a(Point{{1., 2., 3.}}, Point{{4., 5., 6.}});
    SegmentMath b(Point{{1., 2., 3.}}, Point{{4., 5., 6.}});
    assert(a == b);
  }
  // != operator
  {
    SegmentMath a(Point{{1., 2., 3.}}, Point{{4., 5., 6.}});
    SegmentMath b(Point{{2., 3., 5.}}, Point{{7., 11., 13.}});
    assert(a != b);
  }
  {
    SegmentMath a(Point{{1., 2., 3.}}, Point{{4., 5., 6.}});
    SegmentMath b(Point{{1., 2., 3.}}, Point{{4., 5., 6.}});
    assert(!(a != b));
  }
  {
    // unary + operator
    SegmentMath s(Point{{1., 2., 3.}}, Point{{2., 3., 5.}});
    assert(s == +s);
  }
  {
    // unary - operator
    SegmentMath s(Point{{1., 2., 3.}}, Point{{2., 3., 5.}});
    SegmentMath t(Point{{2., 3., 5.}}, Point{{1., 2., 3.}});
    assert(s == -t);
  }
  {
    // distance
    SegmentMath s(Point{{0., 0., 0.}}, Point{{1., 0., 0.}});
    Point p;

    p = Point{{0., 0., 0.}};
    assert(std::abs(geom::computeDistance(s, p) - 0) < 1e-8);

    p = Point{{1., 0., 0.}};
    assert(std::abs(geom::computeDistance(s, p) - 0) < 1e-8);

    p = Point{{0.5, 0., 0.}};
    assert(std::abs(geom::computeDistance(s, p) - 0) < 1e-8);

    p = Point{{0., 0., 1.}};
    assert(std::abs(geom::computeDistance(s, p) - 1) < 1e-8);

    p = Point{{0.5, 0., 1.}};
    assert(std::abs(geom::computeDistance(s, p) - 1) < 1e-8);

    p = Point{{-1., 0., 0.}};
    assert(std::abs(geom::computeDistance(s, p) - 1) < 1e-8);

    p = Point{{2., 1., 0.}};
    assert(std::abs(geom::computeDistance(s, p) - sqrt(2.0)) < 1e-8);
  }
  {
    // closest point
    SegmentMath s(Point{{0., 0., 0.}}, Point{{1., 0., 0.}});
    Point p, cp;

    p = Point{{0., 0., 0.}};
    assert(std::abs(geom::computeDistanceAndClosestPoint(s, p, &cp) - 0) <
           1e-8);
    assert(stlib::ext::euclideanDistance(cp, Point{{0., 0., 0.}}) < 1e-8);

    p = Point{{1., 0., 0.}};
    assert(std::abs(geom::computeDistanceAndClosestPoint(s, p, &cp) - 0) <
           1e-8);
    assert(stlib::ext::euclideanDistance(cp, Point{{1., 0., 0.}}) < 1e-8);

    p = Point{{0.5, 0., 0.}};
    assert(std::abs(geom::computeDistanceAndClosestPoint(s, p, &cp) - 0) <
           1e-8);
    assert(stlib::ext::euclideanDistance(cp, Point{{0.5, 0., 0.}}) < 1e-8);

    p = Point{{0., 0., 1.}};
    assert(std::abs(geom::computeDistanceAndClosestPoint(s, p, &cp) - 1) <
           1e-8);
    assert(stlib::ext::euclideanDistance(cp, Point{{0., 0., 0.}}) < 1e-8);

    p = Point{{0.5, 0., 1.}};
    assert(std::abs(geom::computeDistanceAndClosestPoint(s, p, &cp) - 1) <
           1e-8);
    assert(stlib::ext::euclideanDistance(cp, Point{{0.5, 0., 0.}}) < 1e-8);

    p = Point{{-1., 0., 0.}};
    assert(std::abs(geom::computeDistanceAndClosestPoint(s, p, &cp) - 1) <
           1e-8);
    assert(stlib::ext::euclideanDistance(cp, Point{{0., 0., 0.}}) < 1e-8);

    p = Point{{2., 1., 0.}};
    assert(std::abs(geom::computeDistanceAndClosestPoint(s, p, &cp) -
                    sqrt(2.0)) < 1e-8);
    assert(stlib::ext::euclideanDistance(cp, Point{{1., 0., 0.}}) < 1e-8);
  }
  {
    // distance supporting line
    SegmentMath s(Point{{0., 0., 0.}}, Point{{1., 0., 0.}});
    Point p;

    p = Point{{0., 0., 0.}};
    assert(std::abs(geom::computeUnsignedDistanceToSupportingLine(s,
                    p) - 0) < 1e-8);

    p = Point{{1., 0., 0.}};
    assert(std::abs(geom::computeUnsignedDistanceToSupportingLine(s,
                    p) - 0) < 1e-8);

    p = Point{{0.5, 0., 0.}};
    assert(std::abs(geom::computeUnsignedDistanceToSupportingLine(s,
                    p) - 0) < 1e-8);

    p = Point{{0., 0., 1.}};
    assert(std::abs(geom::computeUnsignedDistanceToSupportingLine(s,
                    p) - 1) < 1e-8);

    p = Point{{0.5, 0., 1.}};
    assert(std::abs(geom::computeUnsignedDistanceToSupportingLine(s,
                    p) - 1) < 1e-8);

    p = Point{{-1., 0., 0.}};
    assert(std::abs(geom::computeUnsignedDistanceToSupportingLine(s,
                    p) - 0) < 1e-8);

    p = Point{{2., 1., 0.}};
    assert(std::abs(geom::computeUnsignedDistanceToSupportingLine(s,
                    p) - 1) < 1e-8);
  }
  {
    // closest point supporting line
    SegmentMath s(Point{{0., 0., 0.}}, Point{{1., 0., 0.}});
    Point p, cp;

    p = Point{{0., 0., 0.}};
    assert(std::abs(geom::computeUnsignedDistanceAndClosestPointToSupportingLine(s,
                    p, &cp) - 0) < 1e-8);
    assert(stlib::ext::euclideanDistance(cp, Point{{0., 0., 0.}}) < 1e-8);

    p = Point{{1., 0., 0.}};
    assert(std::abs(geom::computeUnsignedDistanceAndClosestPointToSupportingLine(s,
                    p, &cp) - 0) < 1e-8);
    assert(stlib::ext::euclideanDistance(cp, Point{{1., 0., 0.}}) < 1e-8);

    p = Point{{0.5, 0., 0.}};
    assert(std::abs(geom::computeUnsignedDistanceAndClosestPointToSupportingLine(s,
                    p, &cp) - 0) < 1e-8);
    assert(stlib::ext::euclideanDistance(cp, Point{{0.5, 0., 0.}}) < 1e-8);

    p = Point{{0., 0., 1.}};
    assert(std::abs(geom::computeUnsignedDistanceAndClosestPointToSupportingLine(s,
                    p, &cp) - 1) < 1e-8);
    assert(stlib::ext::euclideanDistance(cp, Point{{0., 0., 0.}}) < 1e-8);

    p = Point{{0.5, 0., 1.}};
    assert(std::abs(geom::computeUnsignedDistanceAndClosestPointToSupportingLine(s,
                    p, &cp) - 1) < 1e-8);
    assert(stlib::ext::euclideanDistance(cp, Point{{0.5, 0., 0.}}) < 1e-8);

    p = Point{{-1., 0., 0.}};
    assert(std::abs(geom::computeUnsignedDistanceAndClosestPointToSupportingLine(s,
                    p, &cp) - 0) < 1e-8);
    assert(stlib::ext::euclideanDistance(cp, Point{{-1., 0., 0.}}) < 1e-8);

    p = Point{{2., 1., 0.}};
    assert(std::abs(geom::computeUnsignedDistanceAndClosestPointToSupportingLine(s,
                    p, &cp) - 1) < 1e-8);
    assert(stlib::ext::euclideanDistance(cp, Point{{2., 0., 0.}}) < 1e-8);
  }
  {
    // z intersect
    SegmentMath s(Point{{0., 0., 0.}}, Point{{1., 1., 1.}});
    double x, y, z = 0.5;
    assert(geom::computeZIntersection(s, &x, &y, z));
    assert(std::abs(x - 0.5) < 1e-8);
    assert(std::abs(y - 0.5) < 1e-8);
  }
  return 0;
}
