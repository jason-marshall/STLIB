// -*- C++ -*-

#include "stlib/levelSet/IntersectionCircle.h"
#include "stlib/numerical/equality.h"

using namespace stlib;
using numerical::areEqual;

int
main()
{
  //
  // makeBoundaryIntersection()
  //
  // 3-D
  {
    typedef float T;
    const std::size_t D = 3;
    typedef std::array<T, D> Point;
    typedef geom::Ball<T, D> Ball;
    typedef geom::Circle3<T> Circle;
    const T Eps = std::numeric_limits<T>::epsilon();

    {
      const Ball a = {{{0, 0, 0}}, 1};
      const Ball b = {{{1, 0, 0}}, 1};
      Circle circle;
      assert(levelSet::makeBoundaryIntersection(a, b, &circle));
      assert(areEqual(circle.center, Point{{0.5, 0, 0}}));
      assert(areEqual(circle.normal, Point{{1, 0, 0}}));
      assert(areEqual(circle.radius, T(0.5 * std::sqrt(3.))));
    }
    {
      // Barely intersect.
      const Ball a = {{{0, 0, 0}}, T(0.5 * (1 + Eps))};
      const Ball b = {{{1, 0, 0}}, T(0.5 * (1 + Eps))};
      Circle circle;
      assert(levelSet::makeBoundaryIntersection(a, b, &circle));
      assert(areEqual(circle.center, Point{{0.5, 0, 0}}));
      assert(areEqual(circle.normal, Point{{1, 0, 0}}));
      // std::sqrt(0.25 * (1 + Eps) * (1 + Eps) - 0.25)
      const T H = std::sqrt(0.25 * (2 * Eps + Eps * Eps));
      assert(areEqual(circle.radius, H, T(1e4)));
    }
    {
      // Don't intersect enough.
      const Ball a = {{{0, 0, 0}}, T(0.5 * (1 + Eps))};
      const Ball b = {{{1, 0, 0}}, 0.5};
      Circle circle;
      assert(! levelSet::makeBoundaryIntersection(a, b, &circle));
    }
    {
      // Don't intersect.
      const Ball a = {{{0, 0, 0}}, T(0.5 * (1 - Eps))};
      const Ball b = {{{1, 0, 0}}, 0.5};
      Circle circle;
      assert(! levelSet::makeBoundaryIntersection(a, b, &circle));
    }
  }

  //
  // boundNegativeDistance()
  //
  // 3-D
  {
    typedef float T;
    const std::size_t D = 3;
    typedef std::array<T, D> Point;
    typedef geom::Ball<T, D> Ball;
    typedef geom::BBox<T, D> BBox;
    typedef geom::Circle3<T> Circle;

    {
      const Ball a = {{{0, 0, 0}}, 1};
      const Ball b = {{{1, 0, 0}}, 1};
      Circle circle;
      assert(levelSet::makeBoundaryIntersection(a, b, &circle));
      BBox box;
      levelSet::boundNegativeDistance(a, b, circle, &box);
      assert(isInside(box, Point{{0, 0, 0}}));
      assert(isInside(box, Point{{1, 0, 0}}));
      assert(isInside(box, Point{{T(0.5), 1 / std::sqrt(T(2)), 0}}));
      assert(isInside(box, Point{{T(0.5), -1 / std::sqrt(T(2)), 0}}));
      assert(isInside(box, Point{{T(0.5), 0, 1 / std::sqrt(T(2))}}));
      assert(isInside(box, Point{{T(0.5), 0, -1 / std::sqrt(T(2))}}));
      assert(! isInside(box, Point{{T(-0.1), 0, 0}}));
      assert(! isInside(box, Point{{T(1.1), 0, 0}}));
      assert(! isInside(box, Point{{T(0.5), 2, 0}}));
      assert(! isInside(box, Point{{T(0.5), -2, 0}}));
      assert(! isInside(box, Point{{T(0.5), 0, 2}}));
      assert(! isInside(box, Point{{T(0.5), 0, -2}}));
    }
  }

  //
  // distance() to a Circle3.
  //
  {
    using levelSet::distance;
    typedef float T;
    const std::size_t D = 3;
    typedef geom::Circle3<T> Circle;
    typedef Circle::Point Point;
    const T Inf = std::numeric_limits<T>::infinity();

    // Center at origin. Normal along x axis. Unit radius.
    const Circle circle = {{{0, 0, 0}}, {{1, 0, 0}}, 1};
    // No intersecting balls.
    {
      std::vector<geom::Ball<T, D> > balls;
      std::vector<std::size_t> intersecting;
      {
        const Point x = {{0, 0, 0}};
        assert(areEqual(distance(circle, x, balls, intersecting), T(-1)));
      }
      {
        const Point x = {{1, 0, 0}};
        assert(areEqual(distance(circle, x, balls, intersecting),
                        -std::sqrt(T(2))));
      }

      {
        const Point x = {{0, 1.001, 0}};
        assert(distance(circle, x, balls, intersecting) == Inf);
      }
      {
        const Point x = {{0, 2, 0}};
        assert(distance(circle, x, balls, intersecting) == Inf);
      }
      {
        const Point x = {{1, 2, 0}};
        assert(distance(circle, x, balls, intersecting) == Inf);
      }
    }
    // Enclosed by an intersecting ball.
    {
      std::vector<geom::Ball<T, D> > balls;
      balls.push_back(geom::Ball<T, D>{Point{{0, 0, 0}}, T(1.1)});
      std::vector<std::size_t> intersecting(1, 0);
      {
        const Point x = {{0, 0, 0}};
        assert(distance(circle, x, balls, intersecting) == -Inf);
      }
      {
        const Point x = {{1, 0, 0}};
        assert(distance(circle, x, balls, intersecting) == -Inf);
      }
    }
  }

  return 0;
}
