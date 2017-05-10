// -*- C++ -*-

#include "stlib/levelSet/powerDistance.h"
#include "stlib/levelSet/count.h"
#include "stlib/numerical/equality.h"

USING_STLIB_EXT_ARRAY_MATH_OPERATORS;
using namespace stlib;
using levelSet::powerDistance;
using levelSet::countKnown;
using levelSet::countUnknown;
using numerical::areEqual;

int
main()
{
  using geom::makeBallSquared;

  // powerDistance()
  {
    typedef double T;
    const std::size_t D = 1;
    const std::size_t N = 11;
    typedef container::EquilateralArray<T, D, N> Array;
    typedef std::array<T, D> Point;
    typedef geom::BallSquared<T, D> Ball;

    const Point LowerCorner = {{0}};
    const T Spacing = 0.1;
    {
      Array grid(std::numeric_limits<T>::quiet_NaN());
      std::vector<Ball> balls(1);
      balls[0] = makeBallSquared(Point{{-2}}, T(1));
      powerDistance(&grid, LowerCorner, Spacing, balls);
      assert(countKnown(grid.begin(), grid.end()) == grid.size());
      assert(countUnknown(grid.begin(), grid.end()) == 0);
      const Ball& b = balls[0];
      for (std::size_t i = 0; i != grid.size(); ++i) {
        const Point x = LowerCorner + i * Spacing;
        const T d = stlib::ext::squaredDistance(x, b.center) - b.squaredRadius;
        assert(areEqual(grid[i], d));
      }
    }
  }

  return 0;
}
