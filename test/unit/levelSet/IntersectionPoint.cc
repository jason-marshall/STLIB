// -*- C++ -*-

#include "stlib/levelSet/IntersectionPoint.h"

#include "stlib/numerical/equality.h"

using namespace stlib;
using numerical::areEqual;

int
main()
{
  //
  // makeBoundaryIntersection()
  //
  // 2-D
  {
    typedef float T;
    const std::size_t D = 2;
    typedef std::array<T, D> Point;
    typedef geom::Ball<T, D> Ball;
    typedef levelSet::IntersectionPoint<T, D> IntersectionPoint;
    const T Eps = std::numeric_limits<T>::epsilon();

    {
      const Ball a = {{{0, 0}}, 1};
      const Ball b = {{{1, 0}}, 1};
      IntersectionPoint p, q;
      assert(makeBoundaryIntersection(a, b, &p, &q));
      assert(areEqual(p.location,
                      Point{{T(0.5), T(0.5 * std::sqrt(3))}}));
      assert(areEqual(q.location,
                      Point{{T(0.5), T(-0.5 * std::sqrt(3))}}));
      assert(areEqual(p.normal, Point{{0, 1}}));
      assert(areEqual(q.normal, Point{{0, -1}}));
      assert(p.radius == a.radius);
      assert(q.radius == a.radius);
    }
    {
      // Barely intersect.
      const Ball a = {{{0, 0}}, T(0.5 * (1 + Eps))};
      const Ball b = {{{1, 0}}, T(0.5 * (1 + Eps))};
      IntersectionPoint p, q;
      assert(makeBoundaryIntersection(a, b, &p, &q));
      // std::sqrt(0.25 * (1 + Eps) * (1 + Eps) - 0.25)
      const T H = std::sqrt(0.25 * (2 * Eps + Eps * Eps));
      assert(areEqual(p.location, Point{{0.5, H}}, T(1e4)));
      assert(areEqual(q.location, Point{{0.5, -H}}, T(1e4)));
      assert(areEqual(p.normal, Point{{0, 1}}));
      assert(areEqual(q.normal, Point{{0, -1}}));
      assert(p.radius == a.radius);
      assert(q.radius == a.radius);
    }
    {
      // Don't intersect enough.
      const Ball a = {{{0, 0}}, T(0.5 * (1 + Eps))};
      const Ball b = {{{1, 0}}, 0.5};
      IntersectionPoint p, q;
      assert(! makeBoundaryIntersection(a, b, &p, &q));
    }
    {
      // Don't intersect.
      const Ball a = {{{0, 0}}, T(0.5 * (1 - Eps))};
      const Ball b = {{{1, 0}}, 0.5};
      IntersectionPoint p, q;
      assert(! makeBoundaryIntersection(a, b, &p, &q));
    }
  }
  // 3-D, Points.
  {
    typedef float T;
    const std::size_t D = 3;
    typedef std::array<T, D> Point;
    typedef geom::Ball<T, D> Ball;
    typedef levelSet::IntersectionPoint<T, D> IntersectionPoint;

    {
      const Ball a = {{{0, 0, 0}}, 1};
      const Ball b = {{{1, 0, 0}}, 1};
      const Ball c = {{{0.5, std::sqrt(T(3)) / 2, 0}}, 1};
      IntersectionPoint p, q;
      assert(makeBoundaryIntersection(a, b, c, &p, &q));
      assert(areEqual(p.location,
                      Point{{T(0.5), T(0.5 / std::sqrt(3)),
                            T(std::sqrt(2. / 3))}}));
      assert(areEqual(q.location,
                      Point{{T(0.5), T(0.5 / std::sqrt(3)),
                            T(-std::sqrt(2. / 3))}}));
      assert(areEqual(p.normal, Point{{0, 0, 1}}));
      assert(areEqual(q.normal, Point{{0, 0, -1}}));
      assert(p.radius == a.radius);
      assert(q.radius == a.radius);
    }
  }

  return 0;
}
