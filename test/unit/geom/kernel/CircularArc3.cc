// -*- C++ -*-

#include "stlib/geom/kernel/CircularArc3.h"

#include <iostream>

#include <cassert>

USING_STLIB_EXT_ARRAY_MATH_OPERATORS;
using namespace stlib;

int
main()
{
  typedef geom::Circle3<double> Circle;
  typedef geom::CircularArc3<> CircularArc;
  typedef CircularArc::Point Point;

  const double Epsilon = std::numeric_limits<double>::epsilon();

  {
    // Default constructor
    CircularArc x;
  }
  {
    // Point constructor
    const Point center = {{1, 2, 3}};
    const Point source = {{2, 2, 3}};
    const Point target = {{1, 3, 3}};
    const CircularArc x(center, source, target);
    assert(x.isValid());
    std::cout << "CircularArc((1,2,3),(2,2,3),(1,3,3)) =\n" << x << "\n";

    // copy constructor
    const CircularArc y(x);
    assert(y == x);

    // assignment operator
    CircularArc z;
    z = x;
    assert(z == x);

    // Accessors.
    assert(x.getCenter() == center);
    assert(std::abs(x.getRadius() - 1.0) < 10.0 * Epsilon);
    assert(stlib::ext::euclideanDistance(x.getFirstAxis(), Point{{1., 0., 0.}}) <
           10.0 * Epsilon);
    assert(stlib::ext::euclideanDistance(x.getSecondAxis(), Point{{0., 1., 0.}}) <
           10.0 * Epsilon);
    assert(std::abs(x.getAngle() - 0.5 * numerical::Constants<double>::Pi())
           < 10.0 * Epsilon);
    assert(stlib::ext::euclideanDistance(x(0.0), source) < 10.0 * Epsilon);
    assert(stlib::ext::euclideanDistance(x(1.0), target) < 10.0 * Epsilon);
  }
  // += operator
  {
    const Point center = {{1, 2, 3}};
    const Point source = {{2, 2, 3}};
    const Point target = {{1, 3, 3}};
    CircularArc x(center, source, target);
    const Point p = {{2, 3, 5}};
    x += p;
    const CircularArc y(center + p, source + p, target + p);
    assert(x == y);
  }
  // -= operator
  {
    const Point center = {{1, 2, 3}};
    const Point source = {{2, 2, 3}};
    const Point target = {{1, 3, 3}};
    CircularArc x(center, source, target);
    const Point p = {{2, 3, 5}};
    x -= p;
    const CircularArc y(center - p, source - p, target - p);
    assert(x == y);
  }
  // == operator
  {
    CircularArc a(Point{{1., 2., 3.}}, Point{{2., 2., 3.}},
                  Point{{1., 3., 3.}});
    CircularArc b(Point{{1., 2., 3.}}, Point{{2., 2., 3.}},
                  Point{{1., 3., 3.}});
    assert(a == b);
  }
  {
    CircularArc a(Point{{1., 2., 3.}}, Point{{2., 2., 3.}},
                  Point{{1., 3., 3.}});
    CircularArc b(Point{{2., 2., 3.}}, Point{{3., 2., 3.}},
                  Point{{2., 3., 3.}});
    assert(!(a == b));
  }
  // != operator
  {
    CircularArc a(Point{{1., 2., 3.}}, Point{{2., 2., 3.}},
                  Point{{1., 3., 3.}});
    CircularArc b(Point{{1., 2., 3.}}, Point{{2., 2., 3.}},
                  Point{{1., 3., 3.}});
    assert(!(a != b));
  }
  {
    CircularArc a(Point{{1., 2., 3.}}, Point{{2., 2., 3.}},
                  Point{{1., 3., 3.}});
    CircularArc b(Point{{2., 2., 3.}}, Point{{3., 2., 3.}},
                  Point{{2., 3., 3.}});
    assert(a != b);
  }
  // Closest point to a circular arc.
  {
    const double Sqrt2 = std::sqrt(2.0);
    const Point center = {{0, 0, 0}};
    const Point source = {{1, 0, 0}};
    const Point target = {{0, 1, 0}};
    const CircularArc x(center, source, target);
    Point cp;

    {
      // Interior of arc.
      const Point ClosestPoint = {{1.0 / Sqrt2, 1.0 / Sqrt2, 0}};

      geom::computeClosestPoint(x, Point{{0.5, 0.5, 0.}}, &cp);
      assert(stlib::ext::euclideanDistance(cp, ClosestPoint) < 10.0 * Epsilon);

      geom::computeClosestPoint(x, Point{{2., 2., 0.}}, &cp);
      assert(stlib::ext::euclideanDistance(cp, ClosestPoint) < 10.0 * Epsilon);

      geom::computeClosestPoint(x, Point{{0.5, 0.5, 1.}}, &cp);
      assert(stlib::ext::euclideanDistance(cp, ClosestPoint) < 10.0 * Epsilon);

      geom::computeClosestPoint(x, Point{{2., 2., 2.}}, &cp);
      assert(stlib::ext::euclideanDistance(cp, ClosestPoint) < 10.0 * Epsilon);
    }

    {
      // Source.
      const Point ClosestPoint = {{1, 0, 0}};

      geom::computeClosestPoint(x, Point{{1., 0., 0.}}, &cp);
      assert(stlib::ext::euclideanDistance(cp, ClosestPoint) < 10.0 * Epsilon);

      geom::computeClosestPoint(x, Point{{0.5, 0., 0.}}, &cp);
      assert(stlib::ext::euclideanDistance(cp, ClosestPoint) < 10.0 * Epsilon);

      geom::computeClosestPoint(x, Point{{2., 0., 0.}}, &cp);
      assert(stlib::ext::euclideanDistance(cp, ClosestPoint) < 10.0 * Epsilon);

      geom::computeClosestPoint(x, Point{{1., -1., 0.}}, &cp);
      assert(stlib::ext::euclideanDistance(cp, ClosestPoint) < 10.0 * Epsilon);

      geom::computeClosestPoint(x, Point{{2., -1., 0.}}, &cp);
      assert(stlib::ext::euclideanDistance(cp, ClosestPoint) < 10.0 * Epsilon);

      geom::computeClosestPoint(x, Point{{1., 0., 1.}}, &cp);
      assert(stlib::ext::euclideanDistance(cp, ClosestPoint) < 10.0 * Epsilon);

      geom::computeClosestPoint(x, Point{{0.5, 0., 2.}}, &cp);
      assert(stlib::ext::euclideanDistance(cp, ClosestPoint) < 10.0 * Epsilon);

      geom::computeClosestPoint(x, Point{{2., 0., 3.}}, &cp);
      assert(stlib::ext::euclideanDistance(cp, ClosestPoint) < 10.0 * Epsilon);

      geom::computeClosestPoint(x, Point{{1., -1., 4.}}, &cp);
      assert(stlib::ext::euclideanDistance(cp, ClosestPoint) < 10.0 * Epsilon);

      geom::computeClosestPoint(x, Point{{2., -1., 5.}}, &cp);
      assert(stlib::ext::euclideanDistance(cp, ClosestPoint) < 10.0 * Epsilon);
    }
    {
      // Target.
      const Point ClosestPoint = {{0, 1, 0}};

      geom::computeClosestPoint(x, Point{{0., 1., 0.}}, &cp);
      assert(stlib::ext::euclideanDistance(cp, ClosestPoint) < 10.0 * Epsilon);

      geom::computeClosestPoint(x, Point{{0., 0.5, 0.}}, &cp);
      assert(stlib::ext::euclideanDistance(cp, ClosestPoint) < 10.0 * Epsilon);

      geom::computeClosestPoint(x, Point{{0., 2., 0.}}, &cp);
      assert(stlib::ext::euclideanDistance(cp, ClosestPoint) < 10.0 * Epsilon);

      geom::computeClosestPoint(x, Point{{-1., 1., 0.}}, &cp);
      assert(stlib::ext::euclideanDistance(cp, ClosestPoint) < 10.0 * Epsilon);

      geom::computeClosestPoint(x, Point{{-1., 2., 0.}}, &cp);
      assert(stlib::ext::euclideanDistance(cp, ClosestPoint) < 10.0 * Epsilon);

      geom::computeClosestPoint(x, Point{{0., 1., 1.}}, &cp);
      assert(stlib::ext::euclideanDistance(cp, ClosestPoint) < 10.0 * Epsilon);

      geom::computeClosestPoint(x, Point{{0., 0.5, 2.}}, &cp);
      assert(stlib::ext::euclideanDistance(cp, ClosestPoint) < 10.0 * Epsilon);

      geom::computeClosestPoint(x, Point{{0., 2., 3.}}, &cp);
      assert(stlib::ext::euclideanDistance(cp, ClosestPoint) < 10.0 * Epsilon);

      geom::computeClosestPoint(x, Point{{-1., 1., 4.}}, &cp);
      assert(stlib::ext::euclideanDistance(cp, ClosestPoint) < 10.0 * Epsilon);

      geom::computeClosestPoint(x, Point{{-1., 2., 5.}}, &cp);
      assert(stlib::ext::euclideanDistance(cp, ClosestPoint) < 10.0 * Epsilon);
    }
  }
  // Closest point to a circle from a circular arc.
  {
    const double Sqrt2 = std::sqrt(2.0);
    Point cp;

    {
      // Interior of arc.
      const Point center = {{0, 0, 0}};
      const Point source = {{1.0 / Sqrt2, 1.0 / Sqrt2, 0}};
      const Point target = {{ -1.0 / Sqrt2, 1.0 / Sqrt2, 0}};
      const CircularArc arc(center, source, target);
      const Point normal = {{1, 0, 0}};
      const double radius = 1;
      const Circle circle = {center, normal, radius};

      const Point ClosestPoint = {{0, 1, 0}};
      geom::computeClosestPoint(circle, arc, &cp);
      assert(stlib::ext::euclideanDistance(cp, ClosestPoint) < std::sqrt(Epsilon));
    }

    {
      // Interior of arc.
      const Point center = {{0, 0, 0}};
      const Point source = {{1.0 / Sqrt2, 1.0 / Sqrt2, 0}};
      const Point target = {{ -1.0 / Sqrt2, 1.0 / Sqrt2, 0}};
      const CircularArc arc(center, source, target);
      const Point normal = {{1, 0, 0}};
      const double radius = 0.5;
      const Circle circle = {center, normal, radius};

      const Point ClosestPoint = {{0, 0.5, 0}};
      geom::computeClosestPoint(circle, arc, &cp);
      assert(stlib::ext::euclideanDistance(cp, ClosestPoint) < std::sqrt(Epsilon));
    }

    {
      // Interior of arc.
      const Point center = {{0, 0, 0}};
      const Point source = {{1.0 / Sqrt2, 1.0 / Sqrt2, 0}};
      const Point target = {{ -1.0 / Sqrt2, 1.0 / Sqrt2, 0}};
      const CircularArc arc(center, source, target);
      const Point normal = {{1, 0, 0}};
      const double radius = 2;
      const Circle circle = {center, normal, radius};

      const Point ClosestPoint = {{0, 2, 0}};
      geom::computeClosestPoint(circle, arc, &cp);
      assert(stlib::ext::euclideanDistance(cp, ClosestPoint) < std::sqrt(Epsilon));
    }

    {
      // Endpoint of arc.
      const Point center = {{0, 0, 0}};
      const Point source = {{1, 0, 0}};
      const Point target = {{0, 1, 0}};
      const CircularArc arc(center, source, target);
      const Point normal = {{1, 0, 0}};
      const double radius = 1;
      const Circle circle = {center, normal, radius};

      const Point ClosestPoint = {{0, 1, 0}};
      geom::computeClosestPoint(circle, arc, &cp);
      assert(stlib::ext::euclideanDistance(cp, ClosestPoint) < std::sqrt(Epsilon));
    }

    {
      // Endpoint of arc.
      const Point center = {{0, 0, 0}};
      const Point source = {{1, 0, 0}};
      const Point target = {{0, 1, 0}};
      const CircularArc arc(center, source, target);
      const Point normal = {{1, 0, 0}};
      const double radius = 0.5;
      const Circle circle = {center, normal, radius};

      const Point ClosestPoint = {{0, 0.5, 0}};
      geom::computeClosestPoint(circle, arc, &cp);
      assert(stlib::ext::euclideanDistance(cp, ClosestPoint) < std::sqrt(Epsilon));
    }

    {
      // Endpoint of arc.
      const Point center = {{0, 0, 0}};
      const Point source = {{1, 0, 0}};
      const Point target = {{0, 1, 0}};
      const CircularArc arc(center, source, target);
      const Point normal = {{1, 0, 0}};
      const double radius = 2;
      const Circle circle = {center, normal, radius};

      const Point ClosestPoint = {{0, 2, 0}};
      geom::computeClosestPoint(circle, arc, &cp);
      assert(stlib::ext::euclideanDistance(cp, ClosestPoint) < std::sqrt(Epsilon));
    }

    {
      // Endpoint of arc.
      const Point center = {{1, 0, 0}};
      const Point source = {{2, 0, 0}};
      const Point target = {{1, 1, 0}};
      const CircularArc arc(center, source, target);
      const Point normal = {{1, 0, 0}};
      const double radius = 1;
      const Circle circle = {Point{{0., 0., 0.}}, normal, radius};

      const Point ClosestPoint = {{0, 1, 0}};
      geom::computeClosestPoint(circle, arc, &cp);
      assert(stlib::ext::euclideanDistance(cp, ClosestPoint) < std::sqrt(Epsilon));
    }
  }

  return 0;
}
