// -*- C++ -*-

#include "stlib/levelSet/negativeDistance.h"
#include "stlib/levelSet/count.h"
#include "stlib/numerical/equality.h"

using namespace stlib;
using levelSet::negativeDistance;
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
  // negativeDistance()
  //
  // 2-D
  {
    typedef float T;
    const std::size_t D = 2;
    const std::size_t N = 8;
    typedef levelSet::GridUniform<T, D> GridUniform;
    typedef GridUniform::IndexList IndexList;
    typedef levelSet::Grid<T, D, N> Grid;
    typedef Grid::VertexPatch VertexPatch;
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
      negativeDistance(&grid, balls);
      IndexList i;
      for (i[0] = 0; i[0] != grid.extents()[0]; ++i[0]) {
        for (i[1] = 0; i[1] != grid.extents()[1]; ++i[1]) {
          const Point x = grid.indexToLocation(i);
          const T d = stlib::ext::euclideanDistance(balls[0].center, x) - 0.5;
          if (d < -Eps) {
            assert(areEqual(grid(i), d, T(10.)));
          }
          else if (d > Eps) {
            assert(grid(i) != grid(i));
          }
        }
      }
    }
    {
      // Grid with one patch that intersects the ball.
      const BBox domain = {{{0, 0}}, {{1, 1}}};
      Grid grid(domain, 1.);
      assert(grid.extents() == ext::filled_array<IndexList>(1));
      assert(grid.lowerCorner == domain.lower);
      assert(areEqual(grid.spacing, T(1. / (N - 1))));
      assert(areEqual(grid.getVoxelPatchLength(), T(N) / (N - 1)));
      assert(grid.numRefined() == 0);
      const T delta = grid.spacing;
      std::vector<Ball> balls(1);
      balls[0] = Ball{{{0.5, 0.5}}, T(0.5)};
      negativeDistance(&grid, balls);
      assert(grid.numRefined() == 1);
      const VertexPatch& patch = grid[0];
      for (std::size_t i = 0; i != patch.extents()[0]; ++i) {
        for (std::size_t j = 0; j != patch.extents()[1]; ++j) {
          const Point x = {{delta * i, delta * j}};
          const T d = stlib::ext::euclideanDistance(balls[0].center, x) - 0.5;
          if (d < -Eps) {
            assert(areEqual(patch(i, j), d, T(10.)));
          }
          else if (d > Eps) {
            assert(patch(i, j) != patch(i, j));
          }
        }
      }
    }
    {
      // Grid that does not intersect the ball.
      const BBox domain = {{{2, 2}}, {{3, 3}}};
      Grid grid(domain, 1.);
      assert(grid.extents() == ext::filled_array<IndexList>(1));
      assert(grid.lowerCorner == domain.lower);
      assert(areEqual(grid.spacing, T(1. / (N - 1))));
      assert(areEqual(grid.getVoxelPatchLength(), T(N) / (N - 1)));
      assert(grid.numRefined() == 0);
      std::vector<Ball> balls(1);
      balls[0] = Ball{{{0.5, 0.5}}, T(0.5)};
      negativeDistance(&grid, balls);
      assert(grid.numRefined() == 0);
      const VertexPatch& patch = grid[0];
      assert(patch.fillValue != patch.fillValue);
    }
    {
      // Grid with 2x2 array of patches that intersect the ball.
      const BBox domain = {{{0, 0}}, {{1, 1}}};
      Grid grid(domain, 1. / (2 * N - 1) * 1.01);
      assert(grid.extents() == ext::filled_array<IndexList>(2));
      assert(grid.lowerCorner == domain.lower);
      assert(areEqual(grid.spacing, T(1. / (2 * N - 1))));
      assert(areEqual(grid.getVoxelPatchLength(), T(2 * N) / (2 * N - 1) / 2));
      assert(grid.numRefined() == 0);
      std::vector<Ball> balls(1);
      balls[0] = Ball{{{0.5, 0.5}}, T(0.5)};
      negativeDistance(&grid, balls);
      assert(grid.numRefined() == 4);
      IndexList p, i;
      for (p[0] = 0; p[0] != grid.extents()[0]; ++p[0]) {
        for (p[1] = 0; p[1] != grid.extents()[1]; ++p[1]) {
          const VertexPatch& patch = grid(p);
          for (i[0] = 0; i[0] != patch.extents()[0]; ++i[0]) {
            for (i[1] = 0; i[1] != patch.extents()[1]; ++i[1]) {
              const Point x = grid.indexToLocation(p, i);
              const T d = stlib::ext::euclideanDistance(balls[0].center, x) -
                0.5;
              if (d < -Eps) {
                assert(areEqual(patch(i), d, T(100.)));
              }
              else if (d > Eps) {
                assert(patch(i) != patch(i));
              }
            }
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
      negativeDistance(&grid, balls);
      IndexList i;
      for (i[0] = 0; i[0] != grid.extents()[0]; ++i[0]) {
        for (i[1] = 0; i[1] != grid.extents()[1]; ++i[1]) {
          const Point x = grid.indexToLocation(i);
          const T d =
            std::min(stlib::ext::euclideanDistance(balls[0].center, x) - 0.5,
                     stlib::ext::euclideanDistance(balls[1].center, x) - 0.5);
          if (d < -Eps) {
            assert(areEqual(grid(i), d, T(10.)));
          }
          else if (d > Eps) {
            assert(grid(i) != grid(i));
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
      negativeDistance(&grid, balls);

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
      assert(grid(1, 10) != grid(1, 10));
      assert(grid(9, 10) != grid(9, 10));
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
    typedef GridUniform::IndexList IndexList;
    typedef levelSet::Grid<T, D, N> Grid;
    typedef Grid::VertexPatch VertexPatch;
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
      negativeDistance(&grid, balls);
      // Loop over the grid points.
      const Iterator end = Iterator::end(grid.extents());
      for (Iterator i = Iterator::begin(grid.extents()); i != end; ++i) {
        const Point x = grid.indexToLocation(*i);
        const T d = stlib::ext::euclideanDistance(balls[0].center, x) - 0.5;
        if (d < -Eps) {
          assert(areEqual(grid(*i), d, T(10.)));
        }
        else if (d > Eps) {
          assert(grid(*i) != grid(*i));
        }
      }
    }
    {
      // AMR grid.
      // Grid with one patch that intersects the ball.
      const BBox domain = {{{0, 0, 0}}, {{1, 1, 1}}};
      Grid grid(domain, 1.);
      assert(grid.extents() == ext::filled_array<IndexList>(1));
      assert(grid.lowerCorner == domain.lower);
      assert(areEqual(grid.spacing, T(1. / (N - 1))));
      assert(areEqual(grid.getVoxelPatchLength(), T(N) / (N - 1)));
      assert(grid.numRefined() == 0);
      std::vector<Ball> balls(1);
      balls[0] = Ball{{{0.5, 0.5, 0.5}}, T(0.5)};
      negativeDistance(&grid, balls);
      assert(grid.numRefined() == 1);
      const VertexPatch& patch = grid[0];
      const IndexList patchIndex = {{0, 0, 0}};
      // Loop over the grid points.
      const Iterator end = Iterator::end(patch.extents());
      for (Iterator i = Iterator::begin(patch.extents()); i != end; ++i) {
        const Point x = grid.indexToLocation(patchIndex, *i);
        const T d = stlib::ext::euclideanDistance(balls[0].center, x) - 0.5;
        if (d < -Eps) {
          assert(areEqual(patch(*i), d, T(10.)));
        }
        else if (d > Eps) {
          assert(patch(*i) != patch(*i));
        }
      }
    }
    {
      // Grid that does not intersect the ball.
      const BBox domain = {{{2, 2, 2}}, {{3, 3, 3}}};
      Grid grid(domain, 1.);
      assert(grid.extents() == ext::filled_array<IndexList>(1));
      assert(grid.lowerCorner == domain.lower);
      assert(areEqual(grid.spacing, T(1. / (N - 1))));
      assert(areEqual(grid.getVoxelPatchLength(), T(N) / (N - 1)));
      assert(grid.numRefined() == 0);
      std::vector<Ball> balls(1);
      balls[0] = Ball{{{0.5, 0.5, 0.5}}, T(0.5)};
      negativeDistance(&grid, balls);
      assert(grid.numRefined() == 0);
      const VertexPatch& patch = grid[0];
      assert(patch.fillValue != patch.fillValue);
    }
    {
      // Grid with 2x2x2 array of patches that intersect the ball.
      const BBox domain = {{{0, 0, 0}}, {{1, 1, 1}}};
      Grid grid(domain, 1. / (2 * N - 1) * 1.01);
      assert(grid.extents() == ext::filled_array<IndexList>(2));
      assert(grid.lowerCorner == domain.lower);
      assert(areEqual(grid.spacing, T(1. / (2 * N - 1))));
      assert(areEqual(grid.getVoxelPatchLength(), T(2 * N) / (2 * N - 1) / 2));
      assert(grid.numRefined() == 0);
      std::vector<Ball> balls(1);
      balls[0] = Ball{{{0.5, 0.5, 0.5}}, T(0.5)};
      negativeDistance(&grid, balls);
      assert(grid.numRefined() == 8);
      // Loop over the grid points.
      const Iterator pEnd = Iterator::end(grid.extents());
      for (Iterator p = Iterator::begin(grid.extents()); p != pEnd; ++p) {
        const VertexPatch& patch = grid(*p);
        const Iterator iEnd = Iterator::end(patch.extents());
        for (Iterator i = Iterator::begin(patch.extents()); i != iEnd;
             ++i) {
          const Point x = grid.indexToLocation(*p, *i);
          const T d = stlib::ext::euclideanDistance(balls[0].center, x) - 0.5;
          if (d < -Eps) {
            assert(areEqualAbs(patch(*i), d));
          }
          else if (d > Eps) {
            assert(patch(*i) != patch(*i));
          }
        }
      }
    }
  }

  return 0;
}
