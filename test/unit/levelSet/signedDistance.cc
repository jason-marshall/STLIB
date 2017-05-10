// -*- C++ -*-

#include "stlib/levelSet/signedDistance.h"
#include "stlib/levelSet/count.h"
#include "stlib/numerical/equality.h"

using namespace stlib;
using levelSet::signedDistance;
using levelSet::countKnown;
using levelSet::countUnknown;
using numerical::areEqual;
using numerical::areEqualAbs;

template<typename _T>
_T
mergeNegativeDistances(const _T a, const _T b)
{
  // First deal with NaN's.
  if (a != a) {
    return b;
  }
  if (b != b) {
    return a;
  }
  if (a > 0) {
    return b;
  }
  if (b > 0) {
    return a;
  }
  return std::max(a, b);
}

int
main()
{
  //
  // signedDistance()
  //
  // 2-D
  {
    typedef float T;
    const std::size_t D = 2;
    const std::size_t N = 8;
    typedef levelSet::GridUniform<T, D> GridUniform;
    typedef GridUniform::IndexList IndexList;
    typedef geom::BBox<T, D> BBox;
    typedef std::array<T, D> Point;
    typedef geom::Ball<T, D> Ball;

    const T Eps = std::numeric_limits<T>::epsilon();
    {
      const BBox domain = {{{0, 0}}, {{1, 1}}};
      // N + 1 grid points.
      GridUniform grid(domain, 1. / N * 1.001);
      std::vector<Ball> balls(1);
      balls[0] = Ball{{{0.5, 0.5}}, T(0.5)};
      const T MaxDistance = 0.1;
      signedDistance(&grid, balls, MaxDistance);
      IndexList i;
      for (i[0] = 0; i[0] != grid.extents()[0]; ++i[0]) {
        for (i[1] = 0; i[1] != grid.extents()[1]; ++i[1]) {
          const Point x = grid.indexToLocation(i);
          const T d = stlib::ext::euclideanDistance(balls[0].center, x) - 0.5;
          if (d < MaxDistance - Eps) {
            assert(areEqual(grid(i), d, T(10.)));
          }
          else {
            assert(areEqual(grid(i), d, T(10.)) ||
                   grid(i) == std::numeric_limits<T>::max());
          }
        }
      }
    }
    {
      // N + 1 grid points.
      const BBox domain = {{{0, 0}}, {{1, 1}}};
      GridUniform grid(domain, 1. / N * 1.001);
      std::vector<Ball> balls(2);
      balls[0] = Ball{{{0, 0}}, T(0.5)};
      balls[1] = Ball{{{1, 0}}, T(0.5)};
      const T MaxDistance = 0.1;
      signedDistance(&grid, balls, MaxDistance);
      IndexList i;
      for (i[0] = 0; i[0] != grid.extents()[0]; ++i[0]) {
        for (i[1] = 0; i[1] != grid.extents()[1]; ++i[1]) {
          const Point x = grid.indexToLocation(i);
          const T d =
            std::min(stlib::ext::euclideanDistance(balls[0].center, x) - 0.5,
                     stlib::ext::euclideanDistance(balls[1].center, x) - 0.5);
          if (d < MaxDistance - Eps) {
            assert(areEqual(grid(i), d, T(10.)));
          }
          else {
            assert(areEqual(grid(i), d, T(10.)) ||
                   grid(i) == std::numeric_limits<T>::max());
          }
        }
      }
    }
    {
      const BBox domain = {{{0, 0}}, {{1, 1}}};
      GridUniform grid(domain, 1. / 10 * 1.001);
      std::vector<Ball> balls(2);
      balls[0] = Ball{{{0, 0}}, T(1)};
      balls[1] = Ball{{{1, 0}}, T(1)};
      signedDistance(&grid, balls, T(1));

      Point p = {{0.5, T(0.5)* std::sqrt(T(3))}};
      for (std::size_t i = 0; i != 11; ++i) {
        const Point x = {{T(0.1 * i), 0}};
        const T d = - stlib::ext::euclideanDistance(p, x);
        assert(areEqual(grid(i, 0), d, T(10.)));
      }
      for (std::size_t i = 1; i != 10; ++i) {
        const Point x = {{T(0.1 * i), 0.1}};
        const T d = - stlib::ext::euclideanDistance(p, x);
        assert(areEqual(grid(i, 1), d, T(10.)));
      }
      for (std::size_t i = 2; i != 9; ++i) {
        const Point x = {{T(0.1 * i), 0.2}};
        const T d = - stlib::ext::euclideanDistance(p, x);
        assert(areEqual(grid(i, 2), d, T(10.)));
      }
      for (std::size_t i = 2; i != 9; ++i) {
        const Point x = {{T(0.1 * i), 0.3}};
        const T d = - stlib::ext::euclideanDistance(p, x);
        assert(areEqual(grid(i, 3), d, T(10.)));
      }
      for (std::size_t i = 3; i != 8; ++i) {
        const Point x = {{T(0.1 * i), 0.4}};
        const T d = - stlib::ext::euclideanDistance(p, x);
        assert(areEqual(grid(i, 4), d, T(10.)));
      }
      for (std::size_t i = 3; i != 8; ++i) {
        const Point x = {{T(0.1 * i), 0.5}};
        const T d = - stlib::ext::euclideanDistance(p, x);
        assert(areEqual(grid(i, 5), d, T(10.)));
      }
      for (std::size_t i = 4; i != 7; ++i) {
        const Point x = {{T(0.1 * i), 0.6}};
        const T d = - stlib::ext::euclideanDistance(p, x);
        assert(areEqual(grid(i, 6), d, T(10.)));
      }
      for (std::size_t i = 5; i != 6; ++i) {
        const Point x = {{T(0.1 * i), 0.7}};
        const T d = - stlib::ext::euclideanDistance(p, x);
        assert(areEqual(grid(i, 7), d, T(10.)));
      }
      for (std::size_t i = 5; i != 6; ++i) {
        const Point x = {{T(0.1 * i), 0.8}};
        const T d = - stlib::ext::euclideanDistance(p, x);
        assert(areEqual(grid(i, 8), d, T(10.)));
      }
      assert(areEqual(grid(0, 0), T(-1), T(10.)));
      assert(areEqual(grid(10, 0), T(-1), T(10.)));
    }
  }
  //
  // 3-D
  //
  {
    typedef float T;
    const std::size_t D = 3;
    const std::size_t N = 8;
    typedef levelSet::GridUniform<T, D> GridUniform;
    typedef geom::BBox<T, D> BBox;
    typedef std::array<T, D> Point;
    typedef geom::Ball<T, D> Ball;
    typedef container::SimpleMultiIndexRangeIterator<D> Iterator;
    const T Eps = std::numeric_limits<T>::epsilon();
    {
      // Uniform grid.
      const BBox domain = {{{0, 0, 0}}, {{1, 1, 1}}};
      // N grid points.
      GridUniform grid(domain, 1. / (N - 1) * 1.001);
      std::vector<Ball> balls(1);
      balls[0] = Ball{{{0.5, 0.5, 0.5}}, T(0.5)};
      const T MaxDistance = 0.1;
      signedDistance(&grid, balls, MaxDistance);
      // Loop over the grid points.
      const Iterator end = Iterator::end(grid.extents());
      for (Iterator i = Iterator::begin(grid.extents()); i != end; ++i) {
        const Point x = grid.indexToLocation(*i);
        const T d = stlib::ext::euclideanDistance(balls[0].center, x) - 0.5;
        if (d < MaxDistance - Eps) {
          assert(areEqual(grid(*i), d, T(10.)));
        }
        else {
          assert(areEqual(grid(*i), d, T(10.)) ||
                 grid(*i) == std::numeric_limits<T>::max());
        }
      }
    }
  }

  return 0;
}
