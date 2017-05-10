// -*- C++ -*-


#include "stlib/levelSet/solventExcluded.h"
#include "stlib/levelSet/marchingSimplices.h"
#include "stlib/numerical/equality.h"


using namespace stlib;

int
main()
{
  typedef float T;

  // interpolateCenter()
  {
    // 1-D.
    {
      typedef levelSet::Grid<T, 1, 2> Grid;
      typedef Grid::VertexPatch VertexPatch;
      typedef Grid::BBox BBox;

      const BBox Domain = {{{0}}, {{1}}};
      Grid grid(Domain, 1);
      {
        std::vector<std::size_t> indices(1, 0);
        grid.refine(indices);
      }
      VertexPatch& patch = grid[0];
      patch[0] = 0;
      patch[1] = 1;
      assert(interpolateCenter(patch) == 0.5);
    }
  }

  // solventExcludedIsFarAway()
  {
    // 1-D.
    {
      const std::size_t D = 1;
      const std::size_t N = 8;
      typedef levelSet::Grid<T, D, N> Grid;
      typedef Grid::IndexList IndexList;
      typedef Grid::BBox BBox;

      const BBox Domain = {{{0}}, {{0.875}}};
      Grid grid(Domain, 0.125 + 0.0001);
      {
        std::vector<std::size_t> indices(1, 0);
        grid.refine(indices);
      }
      const IndexList gridIndex = ext::filled_array<IndexList>(0);

      std::fill(grid[0].begin(), grid[0].end(), T(-1));
      assert(! solventExcludedIsFarAway(grid, gridIndex, T(10), T(0)));

      std::fill(grid[0].begin(), grid[0].end(), T(0));
      assert(solventExcludedIsFarAway(grid, gridIndex, T(0.5 + 0.875 / 2),
                                      T(0)));

      grid[0][3] = 0.5;
      assert(solventExcludedIsFarAway(grid, gridIndex, T(0.875 / 2), T(0)));
      grid[0][3] = 0;

      grid[0][0] = 1;
      assert(solventExcludedIsFarAway(grid, gridIndex, T(0), T(0)));
      grid[0][0] = 0;
    }
    // 3-D.
    {
      const std::size_t D = 3;
      const std::size_t N = 8;
      typedef levelSet::Grid<T, D, N> Grid;
      typedef Grid::IndexList IndexList;
      typedef Grid::BBox BBox;

      const BBox Domain = {{{0, 0, 0}}, {{0.875, 0.875, 0.875}}};
      Grid grid(Domain, 0.125 + 0.0001);
      {
        std::vector<std::size_t> indices(1, 0);
        grid.refine(indices);
      }
      const IndexList gridIndex = ext::filled_array<IndexList>(0);

      std::fill(grid[0].begin(), grid[0].end(), T(-1));
      assert(! solventExcludedIsFarAway(grid, gridIndex, T(10), T(0)));

      std::fill(grid[0].begin(), grid[0].end(), T(0));
      assert(solventExcludedIsFarAway(grid, gridIndex,
                                      T(1.01 * 0.875 * std::sqrt(T(D))),
                                      T(0)));

      grid[0][0] = 2;
      assert(solventExcludedIsFarAway(grid, gridIndex, T(0), T(0)));
      grid[0][0] = 0;

      const IndexList center = {{3, 3, 3}};
      grid[0](center) = 1.01 * 0.5 * std::sqrt(T(D));
      assert(solventExcludedIsFarAway(grid, gridIndex, T(0), T(0)));
      grid[0](center) = 0;
    }
  }

  // solventExcludedSeeds()
  {
    // 1-D.
    {
      const std::size_t D = 1;
      const std::size_t N = 8;
      typedef levelSet::GridGeometry<D, N, T> Grid;
      typedef Grid::BBox BBox;
      typedef geom::Ball<T, D> Ball;

      const BBox Domain = {{{ -3}}, {{2.875}}};
      Grid grid(Domain, 0.125 + 0.0001);
      std::vector<Ball> balls(1);
      {
        geom::Ball<T, D> b = {{{0}}, 1};
        balls[0] = b;
      }
      const T ProbeRadius = 1.4;
      std::vector<Ball> seeds;
      solventExcludedSeeds(grid, balls, ProbeRadius, &seeds);
      assert(2 <= seeds.size() && seeds.size() <= 4);
      for (std::size_t i = 0; i != seeds.size(); ++i) {
        assert(numerical::areEqual(stlib::ext::euclideanDistance
                                   (balls[0].center,
                                    seeds[i].center),
                                   balls[0].radius + seeds[i].radius));
      }
    }
  }

  // solventExcludedUsingSeeds()
  {
    // 1-D.
    {
      const std::size_t D = 1;
      typedef levelSet::Grid<T, D, 8> Grid;
      typedef Grid::BBox BBox;

      const BBox Domain = {{{ -3}}, {{2.875}}};
      Grid grid(Domain, 0.125 + 0.0001);
      std::vector<geom::Ball<T, D> > balls(1);
      {
        geom::Ball<T, D> b = {{{0}}, 1};
        balls[0] = b;
      }
      const T ProbeRadius = 1.4;
      solventExcludedUsingSeeds(&grid, balls, ProbeRadius);
      printInfo(grid, std::cout);
      assert(numerical::areEqual(content(grid), T(2)));
    }
  }

  return 0;
}
