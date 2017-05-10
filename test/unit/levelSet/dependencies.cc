// -*- C++ -*-

#include "stlib/levelSet/dependencies.h"

using namespace stlib;

int
main()
{
  typedef float T;

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

      std::vector<Ball> balls(2);
      balls[0].radius = 1;
      balls[1].radius = 1;

      std::vector<std::vector<std::size_t> > dep(1);
      dep[0].push_back(0);
      dep[0].push_back(1);
      container::StaticArrayOfArrays<unsigned> dependencies(dep);

      balls[0].center[0] = 0.5;
      balls[1].center[0] = 0;
      putClosestBallsFirst(grid, balls, &dependencies);
      assert(dependencies(0, 0) == 0);
      assert(dependencies(0, 1) == 1);

      balls[0].center[0] = 0;
      balls[1].center[0] = 0.5;
      putClosestBallsFirst(grid, balls, &dependencies);
      assert(dependencies(0, 0) == 1);
      assert(dependencies(0, 1) == 0);
    }
  }

  return 0;
}
