// -*- C++ -*-

#include "stlib/levelSet/positiveDistance.h"
#include "stlib/levelSet/count.h"
#include "stlib/numerical/equality.h"

using namespace stlib;
using levelSet::positiveDistance;
using levelSet::countKnown;
using levelSet::countUnknown;
using numerical::areEqual;

int
main()
{
  // positiveDistance()
  {
    typedef double T;
    const std::size_t D = 1;
    typedef container::SimpleMultiArray<T, D> MultiArray;
    typedef MultiArray::IndexList IndexList;
    typedef geom::BBox<T, D> BBox;
    typedef geom::Ball<T, D> Ball;

    {
      MultiArray grid(IndexList{{11}},
                      std::numeric_limits<T>::quiet_NaN());
      BBox domain = {{{0}}, {{1}}};
      std::vector<Ball> balls(1);
      balls[0] = Ball{{{-2}}, T(1)};
      const T offset = 0;
      positiveDistance(&grid, domain, balls, offset);
      assert(countKnown(grid.begin(), grid.end()) == 0);
      assert(countUnknown(grid.begin(), grid.end()) == grid.size());
    }
    {
      MultiArray grid(IndexList{{11}},
                      std::numeric_limits<T>::quiet_NaN());
      BBox domain = {{{0}}, {{1}}};
      std::vector<Ball> balls(1);
      balls[0] = Ball{{{4}}, T(1)};
      const T offset = 0;
      positiveDistance(&grid, domain, balls, offset);
      assert(countKnown(grid.begin(), grid.end()) == 0);
      assert(countUnknown(grid.begin(), grid.end()) == grid.size());
    }
    {
      MultiArray grid(IndexList{{11}},
                      std::numeric_limits<T>::quiet_NaN());
      BBox domain = {{{0}}, {{1}}};
      std::vector<Ball> balls(1);
      balls[0] = Ball{{{0.5}}, T(1)};
      const T offset = 0;
      positiveDistance(&grid, domain, balls, offset);
      assert(countKnown(grid.begin(), grid.end()) == grid.size());
      assert(countUnknown(grid.begin(), grid.end()) == 0);
      assert(areEqual(grid[0], -0.5));
      assert(areEqual(grid[5], -1));
      assert(areEqual(grid[10], -0.5));
    }
    {
      MultiArray grid(IndexList{{11}},
                      std::numeric_limits<T>::quiet_NaN());
      BBox domain = {{{0}}, {{1}}};
      std::vector<Ball> balls(1);
      balls[0] = Ball{{{-1}}, T(1)};
      const T offset = 0;
      const T maxDistance = 1.1;
      positiveDistance(&grid, domain, balls, offset, maxDistance);
      assert(countKnown(grid.begin(), grid.end()) == grid.size());
      assert(countUnknown(grid.begin(), grid.end()) == 0);
      assert(areEqual(grid[0], 0));
      assert(areEqual(grid[5], 0.5));
      assert(areEqual(grid[10], 1));
    }
    {
      MultiArray grid(IndexList{{11}},
                      std::numeric_limits<T>::quiet_NaN());
      BBox domain = {{{0}}, {{1}}};
      std::vector<Ball> balls(1);
      balls[0] = Ball{{{-1}}, T(1)};
      const T offset = 0;
      const T maxDistance = 0.51;
      positiveDistance(&grid, domain, balls, offset, maxDistance);
      assert(countKnown(grid.begin(), grid.end()) == 6);
      assert(countUnknown(grid.begin(), grid.end()) == grid.size() - 6);
      assert(areEqual(grid[0], 0));
      assert(areEqual(grid[5], 0.5));
    }
    {
      MultiArray grid(IndexList{{11}},
                      std::numeric_limits<T>::quiet_NaN());
      BBox domain = {{{0}}, {{1}}};
      std::vector<Ball> balls(1);
      balls[0] = Ball{{{0}}, T(1)};
      const T offset = 0;
      const T maxDistance = -0.49;
      positiveDistance(&grid, domain, balls, offset, maxDistance);
      assert(countKnown(grid.begin(), grid.end()) == 6);
      assert(countUnknown(grid.begin(), grid.end()) == grid.size() - 6);
      assert(areEqual(grid[0], -1));
      assert(areEqual(grid[5], -0.5));
    }
  }

  return 0;
}
