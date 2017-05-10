// -*- C++ -*-

#include "stlib/levelSet/solventExcludedCavities.h"

#include "stlib/numerical/equality.h"

using namespace stlib;

int
main()
{
  using numerical::areEqual;

  typedef float T;

  // solventExcludedCavities()
  {
    // 1-D.
    const std::size_t D = 1;
    const T ProbeRadius = 1.4;
    const T TargetGridSpacing = 0.1;
    typedef geom::Ball<T, D> Ball;

    std::vector<Ball> balls;
    {
      Ball b = {{{0}}, 1};
      balls.push_back(b);
    }
    {
      Ball b = {{{3}}, 1};
      balls.push_back(b);
    }

    const std::pair<T, T> content =
      levelSet::solventExcludedCavities(balls, ProbeRadius,
                                        TargetGridSpacing);
    assert(areEqual(content.first, T(1)));
    assert(areEqual(content.second, T(1)));
  }

  // solventExcludedCavitiesPatches()
  {
    // 1-D.
    const std::size_t D = 1;
    const std::size_t N = 8;
    typedef geom::Ball<T, D> Ball;
    typedef levelSet::GridGeometry<D, N, T> Grid;
    typedef Grid::BBox BBox;
    typedef Grid::IndexList IndexList;

    {
      // A single patch.
      const BBox Domain = {{{0}}, {{1}}};
      const Grid grid(Domain, T(1));
      assert(grid.gridExtents == ext::filled_array<IndexList>(1));
      std::vector<Ball> balls;
      {
        Ball ball = {{{0}}, 1};
        balls.push_back(ball);
      }
      const T probeRadius = 1.4;
      std::vector<bool> relevant(1);

      solventExcludedCavitiesPatches(grid, balls, probeRadius, &relevant);
      assert(relevant[0]);

      balls[0].center[0] = -(1 + probeRadius + grid.spacing) * 1.01;
      solventExcludedCavitiesPatches(grid, balls, probeRadius, &relevant);
      assert(! relevant[0]);

      balls[0].center[0] = -(1 + probeRadius + grid.spacing) * 0.99;
      solventExcludedCavitiesPatches(grid, balls, probeRadius, &relevant);
      assert(relevant[0]);

      balls[0].center[0] = 0.5;
      balls[0].radius = (0.5 * (N - 1) + 1) * grid.spacing * 1.01;
      solventExcludedCavitiesPatches(grid, balls, probeRadius, &relevant);
      assert(! relevant[0]);

      balls[0].center[0] = 0.5;
      balls[0].radius = (0.5 * (N - 1) + 1) * grid.spacing * 0.99;
      solventExcludedCavitiesPatches(grid, balls, probeRadius, &relevant);
      assert(relevant[0]);
    }
    {
      // Two patches.
      const BBox Domain = {{{0}}, {{2 - 1. / 8}}};
      const Grid grid(Domain, T(1. / 8 * 1.01));
      assert(grid.gridExtents == ext::filled_array<IndexList>(2));
      std::vector<Ball> balls;
      {
        Ball ball = {{{0}}, 1};
        balls.push_back(ball);
      }
      const T probeRadius = 1.4;
      std::vector<bool> relevant(1);

      solventExcludedCavitiesPatches(grid, balls, probeRadius, &relevant);
      assert(relevant[0] && relevant[1]);

      balls[0].center[0] = -(1 + probeRadius + grid.spacing) * 1.01;
      solventExcludedCavitiesPatches(grid, balls, probeRadius, &relevant);
      assert(! relevant[0] && ! relevant[1]);

      balls[0].center[0] = -(1 + probeRadius + grid.spacing) * 0.99;
      solventExcludedCavitiesPatches(grid, balls, probeRadius, &relevant);
      assert(relevant[0] && ! relevant[1]);

      balls[0].center[0] = 0.5 * (N - 1) * grid.spacing;
      balls[0].radius = (0.5 * (N - 1) + 1) * grid.spacing * 1.01;
      solventExcludedCavitiesPatches(grid, balls, probeRadius, &relevant);
      assert(! relevant[0] && relevant[1]);

      balls[0].center[0] = 0.5 * (N - 1) * grid.spacing;
      balls[0].radius = (0.5 * (N - 1) + 1) * grid.spacing * 0.99;
      solventExcludedCavitiesPatches(grid, balls, probeRadius, &relevant);
      assert(relevant[0] && relevant[1]);
    }
  }

  return 0;
}
