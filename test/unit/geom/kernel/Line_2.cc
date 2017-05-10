// -*- C++ -*-

#include "stlib/geom/kernel/Line_2.h"

#include <iostream>

using namespace stlib;
using namespace geom;

int
main()
{
  typedef Line_2<double> Line;
  typedef Line::Segment Segment;
  typedef Line::Point Point;
  {
    // Default constructor
    std::cout << "Line() = " << '\n'
              << Line() << '\n';
  }
  {
    // Point constructor
    Point source = {{1, 2}}, target = {{1, 3}}, tangent = {{0, 1}},
    normal = {{1, 0}};
    Segment segment(source, target);
    Line ln(source, target);
    std::cout << "Line((1,2),(1,3)) = " << '\n'
              << ln << '\n';

    // Segment constructor
    Line seg(segment);
    assert(seg == ln);
    // copy constructor
    Line copy(ln);
    assert(copy == ln);
    // assignment operator
    Line assign;
    assign = ln;
    assert(assign == ln);

    // accessors
    assert(ln.getPointOn() == source);
    assert(stlib::ext::euclideanDistance(ln.getTangent(), tangent) < 1e-8);
    assert(stlib::ext::euclideanDistance(ln.getNormal(), normal) < 1e-8);

    {
      // Translation
      Line x(ln);
      Point offset = {{5, 7}};
      x += offset;
      assert(x.getPointOn() == source + offset);
      x -= offset;
      assert(x.getPointOn() == source);
    }

    // Distance
    assert(std::abs(ln.computeSignedDistance(Point{{2., 2.}}) - 1)
           < 1e-8);

    // Closest point.
    {
      Point cp;
      double dist = ln.computeSignedDistanceAndClosestPoint(Point{{2., 2.}},
                                                            &cp);
      assert(std::abs(dist - 1) < 1e-8);
      assert(stlib::ext::euclideanDistance(cp, source) < 1e-8);
    }
  }

  return 0;
}
