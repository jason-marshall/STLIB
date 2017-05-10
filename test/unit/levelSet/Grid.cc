// -*- C++ -*-

#include "stlib/levelSet/Grid.h"

#include "stlib/numerical/equality.h"
#include "stlib/geom/kernel/Ball.h"

#include <iostream>
#include <sstream>

using namespace stlib;

template<std::size_t D>
void
test()
{
  using numerical::areEqual;

  typedef double Number;
  typedef std::array<Number, D> Point;
  typedef geom::BBox<Number, D> BBox;
  typedef levelSet::Grid<Number, D, 1> Grid;
  typedef typename Grid::IndexList IndexList;

  std::cout << "\nDimension = " << D << '\n';

  {
    // Grid spacing.
    const Point lower = ext::filled_array<Point>(0);
    const Point upper = ext::filled_array<Point>(1);
    const BBox domain = {lower, upper};
    {
      Grid x(domain, 2);
      assert(x.isValid());
      assert(areEqual(x.spacing, 1));
      assert((x.size() ==
              numerical::Exponentiation<std::size_t, 2, D>::Result));
      assert(x.numRefined() == 0);
      assert(x.numVertices() == 0);
    }
    {
      Grid x(domain, 1.001);
      assert(x.isValid());
      assert(areEqual(x.spacing, 1));
      assert((x.size() ==
              numerical::Exponentiation<std::size_t, 2, D>::Result));
      assert(x.numRefined() == 0);
      assert(x.numVertices() == 0);
    }
    {
      Grid x(domain, 0.5001);
      assert(x.isValid());
      assert(areEqual(x.spacing, 0.5));
      assert((x.size() ==
              numerical::Exponentiation<std::size_t, 3, D>::Result));
      assert(x.numRefined() == 0);
      assert(x.numVertices() == 0);
    }
    {
      Grid x(domain, 0.1001);
      assert(x.isValid());
      assert(areEqual(x.spacing, 0.1));
      assert((x.size() ==
              numerical::Exponentiation<std::size_t, 11, D>::Result));
      assert(x.numRefined() == 0);
      assert(x.numVertices() == 0);
    }

    // +=
    {
      Grid x(domain, 1.001);
      x[0].fillValue = 0;
      x += Number(1);
      assert(x[0].fillValue == 1);
    }

    // Refine a single patch.
    {
      const std::size_t NumPatches =
        numerical::Exponentiation<std::size_t, 11, D>::Result;
      Grid x(domain, 0.1001);
      assert(x.size() == NumPatches);
      assert(x.numRefined() == 0);
      assert(x.numVertices() == 0);
      IndexList i = ext::filled_array<IndexList>(0);

      // Test dual indexing for an unrefined patch.
      x(i, i) = 2;
      assert(x(i, i) == 2);

      std::vector<std::size_t> refine;
      refine.push_back(0);
      x.refine(refine);
      assert(x.isValid());
      assert(x.size() == NumPatches);
      assert(x.numRefined() == 1);
      assert(x.numVertices() == Grid::NumVerticesPerPatch);

      // Test dual indexing for a refined patch.
      const Number old = x(i, i);
      x(i, i) = 2;
      assert(x(i, i) == 2);
      x(i, i) = old;

      x.coarsen();
      assert(x.numRefined() == 0);
    }
  }
}


int
main()
{
  using numerical::areEqual;

  test<1>();
  test<2>();
  test<3>();

  // report(), adjacentNeighbors(), and allNeighbors().
  {
    const std::size_t Dimension = 2;
    const std::size_t N = 4;
    typedef float Number;
    typedef levelSet::Grid<Number, Dimension, N> Grid;
    typedef Grid::IndexList IndexList;
    typedef Grid::Range Range;
    typedef geom::BBox<Number, Dimension> BBox;
    typedef std::pair<IndexList, IndexList> DualIndices;

    BBox domain = {{{0, 0}}, {{1, 1}}};
    Grid grid(domain, Number(1. / (8 - 1) * 1.001));
    assert(grid.extents() == ext::filled_array<IndexList>(2));

    {
      IndexList i = {{0, 0}};
      Range r = {{{1, 4}}, {{0, 0}}};
      std::vector<DualIndices> indices;
      grid.report(i, r, std::back_inserter(indices));
      assert(indices.size() == 4);
      for (std::size_t n = 0; n != 4; ++n) {
        assert(indices[n].first == i);
        assert(indices[n].second == (IndexList{{0, n}}));
      }
    }
    {
      IndexList patch = {{0, 0}};
      IndexList index = {{0, 0}};
      std::vector<DualIndices> neighbors;
      const IndexList NullIndex =
        ext::filled_array<IndexList>
        (std::numeric_limits<std::size_t>::max());

      // All unrefined.

      neighbors.clear();
      grid.adjacentNeighbors(patch, index, 0, std::back_inserter(neighbors));
      assert(neighbors.empty());

      neighbors.clear();
      grid.adjacentNeighbors(patch, index, 1, std::back_inserter(neighbors));
      assert(neighbors.size() == 1);
      assert(neighbors[0].first == (IndexList{{1, 0}}));
      assert(neighbors[0].second == NullIndex);

      neighbors.clear();
      grid.adjacentNeighbors(patch, index, 2, std::back_inserter(neighbors));
      assert(neighbors.empty());

      neighbors.clear();
      grid.adjacentNeighbors(patch, index, 3, std::back_inserter(neighbors));
      assert(neighbors.size() == 1);
      assert(neighbors[0].first == (IndexList{{0, 1}}));
      assert(neighbors[0].second == NullIndex);

      neighbors.clear();
      grid.allNeighbors(std::make_pair(patch, index),
                        std::back_inserter(neighbors));
      assert(neighbors.size() == 2 * 4 + 1);

      // Refined / unrefined.

      {
        // Refine the first patch.
        std::vector<std::size_t> indices(1, 0);
        grid.refine(indices);
      }

      neighbors.clear();
      grid.adjacentNeighbors(patch, index, 0, std::back_inserter(neighbors));
      assert(neighbors.empty());

      neighbors.clear();
      grid.adjacentNeighbors(patch, index, 1, std::back_inserter(neighbors));
      assert(neighbors.size() == 1);
      assert(neighbors[0].first == patch);
      assert(neighbors[0].second == (IndexList{{1, 0}}));

      neighbors.clear();
      grid.adjacentNeighbors(patch, index, 2, std::back_inserter(neighbors));
      assert(neighbors.empty());

      neighbors.clear();
      grid.adjacentNeighbors(patch, index, 3, std::back_inserter(neighbors));
      assert(neighbors.size() == 1);
      assert(neighbors[0].first == patch);
      assert(neighbors[0].second == (IndexList{{0, 1}}));

      neighbors.clear();
      grid.allNeighbors(std::make_pair(patch, index),
                        std::back_inserter(neighbors));
      assert(neighbors.size() == 3);

      index = IndexList{{3, 0}};

      neighbors.clear();
      grid.adjacentNeighbors(patch, index, 0, std::back_inserter(neighbors));
      assert(neighbors.size() == 1);
      assert(neighbors[0].first == patch);
      assert(neighbors[0].second == (IndexList{{2, 0}}));

      neighbors.clear();
      grid.adjacentNeighbors(patch, index, 1, std::back_inserter(neighbors));
      assert(neighbors.size() == 1);
      assert(neighbors[0].first == (IndexList{{1, 0}}));
      assert(neighbors[0].second == NullIndex);

      neighbors.clear();
      grid.adjacentNeighbors(patch, index, 2, std::back_inserter(neighbors));
      assert(neighbors.empty());

      neighbors.clear();
      grid.adjacentNeighbors(patch, index, 3, std::back_inserter(neighbors));
      assert(neighbors.size() == 1);
      assert(neighbors[0].first == patch);
      assert(neighbors[0].second == (IndexList{{3, 1}}));

      neighbors.clear();
      grid.allNeighbors(std::make_pair(patch, index),
                        std::back_inserter(neighbors));
      assert(neighbors.size() == 5);

      index = IndexList{{3, 3}};

      neighbors.clear();
      grid.adjacentNeighbors(patch, index, 0, std::back_inserter(neighbors));
      assert(neighbors.size() == 1);
      assert(neighbors[0].first == patch);
      assert(neighbors[0].second == (IndexList{{2, 3}}));

      neighbors.clear();
      grid.adjacentNeighbors(patch, index, 1, std::back_inserter(neighbors));
      assert(neighbors.size() == 1);
      assert(neighbors[0].first == (IndexList{{1, 0}}));
      assert(neighbors[0].second == NullIndex);

      neighbors.clear();
      grid.adjacentNeighbors(patch, index, 2, std::back_inserter(neighbors));
      assert(neighbors.size() == 1);
      assert(neighbors[0].first == patch);
      assert(neighbors[0].second == (IndexList{{3, 2}}));

      neighbors.clear();
      grid.adjacentNeighbors(patch, index, 3, std::back_inserter(neighbors));
      assert(neighbors.size() == 1);
      assert(neighbors[0].first == (IndexList{{0, 1}}));
      assert(neighbors[0].second == NullIndex);

      neighbors.clear();
      grid.allNeighbors(std::make_pair(patch, index),
                        std::back_inserter(neighbors));
      assert(neighbors.size() == 8);

      // Unrefined / refined.

      patch = IndexList{{1, 0}};
      index = IndexList{{0, 0}};

      neighbors.clear();
      grid.adjacentNeighbors(patch, index, 0, std::back_inserter(neighbors));
      assert(neighbors.size() == 4);
      for (std::size_t i = 0; i != 4; ++i) {
        assert(neighbors[i].first == (IndexList{{0, 0}}));
        assert(neighbors[i].second == (IndexList{{3, i}}));
      }

      neighbors.clear();
      grid.adjacentNeighbors(patch, index, 1, std::back_inserter(neighbors));
      assert(neighbors.empty());

      neighbors.clear();
      grid.adjacentNeighbors(patch, index, 2, std::back_inserter(neighbors));
      assert(neighbors.empty());

      neighbors.clear();
      grid.adjacentNeighbors(patch, index, 3, std::back_inserter(neighbors));
      assert(neighbors.size() == 1);
      assert(neighbors[0].first == (IndexList{{1, 1}}));
      assert(neighbors[0].second == NullIndex);

      neighbors.clear();
      grid.allNeighbors(std::make_pair(patch, index),
                        std::back_inserter(neighbors));
      assert(neighbors.size() == 9);

      // Refined / refined.

      {
        // Refine the first and second patch.
        std::vector<std::size_t> indices(2);
        indices[0] = 0;
        indices[1] = 1;
        grid.refine(indices);
      }

      patch = IndexList{{0, 0}};
      index = IndexList{{3, 0}};

      neighbors.clear();
      grid.adjacentNeighbors(patch, index, 0, std::back_inserter(neighbors));
      assert(neighbors.size() == 1);
      assert(neighbors[0].first == patch);
      assert(neighbors[0].second == (IndexList{{2, 0}}));

      neighbors.clear();
      grid.adjacentNeighbors(patch, index, 1, std::back_inserter(neighbors));
      assert(neighbors.size() == 1);
      assert(neighbors[0].first == (IndexList{{1, 0}}));
      assert(neighbors[0].second == (IndexList{{0, 0}}));

      neighbors.clear();
      grid.adjacentNeighbors(patch, index, 2, std::back_inserter(neighbors));
      assert(neighbors.empty());

      neighbors.clear();
      grid.adjacentNeighbors(patch, index, 3, std::back_inserter(neighbors));
      assert(neighbors.size() == 1);
      assert(neighbors[0].first == patch);
      assert(neighbors[0].second == (IndexList{{3, 1}}));

      neighbors.clear();
      grid.allNeighbors(std::make_pair(patch, index),
                        std::back_inserter(neighbors));
      assert(neighbors.size() == 5);
    }

  }

  // patchDependencies() and getIntersectingPatches()
  // 1-D
  {
    const std::size_t Dimension = 1;
    const std::size_t N = 2;
    typedef float Number;
    typedef levelSet::Grid<Number, Dimension, N> Grid;
    typedef Grid::IndexList IndexList;
    typedef Grid::VoxelPatch VoxelPatch;
    typedef geom::BBox<Number, Dimension> BBox;

    BBox domain = {{{0}}, {{1}}};
    Grid grid(domain, Number(1.001 / 3));
    assert(grid.extents()[0] == 2);
    assert(areEqual(grid.spacing, Number(1. / 3)));

    // Refined, unrefined.
    {
      std::vector<std::size_t> indices(1, 0);
      grid.refine(indices);
      assert(grid[0].isRefined());
      grid[0][0] = 0;
      grid[0][1] = 1;
      VoxelPatch patch;
      IndexList i = {{0}};

      grid.getVoxelPatch(i, &patch);
      assert(patch[0] == 0);
      assert(patch[1] == 1);
      assert(patch[2] == 1);
    }
    // Unrefined, refined.
    {
      std::vector<std::size_t> indices(1, 1);
      grid.refine(indices);
      assert(grid[1].isRefined());
      grid[1][0] = 0;
      grid[1][1] = 1;
      VoxelPatch patch;
      IndexList i = {{1}};

      grid.getVoxelPatch(i, &patch);
      assert(patch[0] == 0);
      assert(patch[1] == 1);
      assert(patch[2] == 1);
    }
    // Refined, refined.
    {
      std::vector<std::size_t> indices(2);
      indices[0] = 0;
      indices[1] = 1;
      grid.refine(indices);
      assert(grid[0].isRefined());
      assert(grid[1].isRefined());
      grid[0][0] = 0;
      grid[0][1] = 1;
      grid[1][0] = 2;
      grid[1][1] = 3;
      VoxelPatch patch;
      IndexList i = {{0}};

      grid.getVoxelPatch(i, &patch);
      assert(patch[0] == 0);
      assert(patch[1] == 1);
      assert(patch[2] == 2);
    }
  }
  // 2-D
  {
    const std::size_t Dimension = 2;
    const std::size_t N = 1;
    typedef float Number;
    typedef levelSet::Grid<Number, Dimension, N> Grid;
    typedef Grid::IndexList IndexList;
    typedef Grid::VoxelPatch VoxelPatch;
    typedef geom::BBox<Number, Dimension> BBox;

    BBox domain = {{{0, 0}}, {{1, 1}}};
    Grid grid(domain, Number(1.001));
    assert(grid.extents() == (IndexList{{2, 2}}));
    assert(areEqual(grid.spacing, Number(1)));

    // uu
    // ru
    {
      std::vector<std::size_t> indices(1, 0);
      grid.refine(indices);
      assert(grid[0].isRefined());
      grid[0][0] = 0;
      VoxelPatch patch;
      IndexList i = {{0, 0}};

      grid.getVoxelPatch(i, &patch);
      assert(patch[0] == 0);
      assert(patch[1] == 0);
      assert(patch[2] == 0);
      assert(patch[3] == 0);
    }
    // uu
    // ur
    {
      const std::size_t n = 1;
      std::vector<std::size_t> indices(1, n);
      grid.refine(indices);
      assert(grid[n].isRefined());
      grid[n][0] = 0;
      VoxelPatch patch;
      IndexList i = {{1, 0}};

      grid.getVoxelPatch(i, &patch);
      assert(patch[0] == 0);
      assert(patch[1] == 0);
      assert(patch[2] == 0);
      assert(patch[3] == 0);
    }
    // ru
    // uu
    {
      const std::size_t n = 2;
      std::vector<std::size_t> indices(1, n);
      grid.refine(indices);
      assert(grid[n].isRefined());
      grid[n][0] = 0;
      VoxelPatch patch;
      IndexList i = {{0, 1}};

      grid.getVoxelPatch(i, &patch);
      assert(patch[0] == 0);
      assert(patch[1] == 0);
      assert(patch[2] == 0);
      assert(patch[3] == 0);
    }
    // ur
    // uu
    {
      const std::size_t n = 3;
      std::vector<std::size_t> indices(1, n);
      grid.refine(indices);
      assert(grid[n].isRefined());
      grid[n][0] = 0;
      VoxelPatch patch;
      IndexList i = {{1, 1}};

      grid.getVoxelPatch(i, &patch);
      assert(patch[0] == 0);
      assert(patch[1] == 0);
      assert(patch[2] == 0);
      assert(patch[3] == 0);
    }
    // rr
    // rr
    {
      std::vector<std::size_t> indices(4);
      for (std::size_t n = 0; n != indices.size(); ++n) {
        indices[n] = n;
      }
      grid.refine(indices);
      for (std::size_t n = 0; n != indices.size(); ++n) {
        grid[n][0] = n;
      }
      VoxelPatch patch;

      {
        IndexList i = {{0, 0}};
        grid.getVoxelPatch(i, &patch);
        assert(patch(0, 0) == 0);
        assert(patch(1, 0) == 1);
        assert(patch(0, 1) == 2);
        assert(patch(1, 1) == 3);
      }
      {
        IndexList i = {{1, 0}};
        grid.getVoxelPatch(i, &patch);
        assert(patch(0, 0) == 1);
        assert(patch(1, 0) == 1);
        assert(patch(0, 1) == 3);
        assert(patch(1, 1) == 1);
      }
      {
        IndexList i = {{0, 1}};
        grid.getVoxelPatch(i, &patch);
        assert(patch(0, 0) == 2);
        assert(patch(1, 0) == 3);
        assert(patch(0, 1) == 2);
        assert(patch(1, 1) == 2);
      }
      {
        IndexList i = {{1, 1}};
        grid.getVoxelPatch(i, &patch);
        assert(patch(0, 0) == 3);
        assert(patch(1, 0) == 3);
        assert(patch(0, 1) == 3);
        assert(patch(1, 1) == 3);
      }
    }
  }
  // 3-D
  {
    const std::size_t Dimension = 3;
    const std::size_t N = 1;
    typedef float Number;
    typedef levelSet::Grid<Number, Dimension, N> Grid;
    typedef Grid::IndexList IndexList;
    typedef Grid::VoxelPatch VoxelPatch;
    typedef geom::BBox<Number, Dimension> BBox;

    BBox domain = {{{0, 0, 0}}, {{1, 1, 1}}};
    Grid grid(domain, Number(1.001));
    assert(grid.extents() == (IndexList{{2, 2, 2}}));
    assert(areEqual(grid.spacing, Number(1)));

    // One refined patch.
    for (std::size_t a = 0; a != 2; ++a) {
      for (std::size_t b = 0; b != 2; ++b) {
        for (std::size_t c = 0; c != 2; ++c) {
          const IndexList i = {{a, b, c}};
          const std::size_t n = i[0] + 2 * i[1] + 4 * i[2];
          std::vector<std::size_t> indices(1, n);
          grid.refine(indices);
          assert(grid[n].isRefined());
          grid[n][0] = 0;
          VoxelPatch patch;

          grid.getVoxelPatch(i, &patch);
          for (std::size_t j = 0; j != patch.size(); ++j) {
            assert(patch[j] == 0);
          }
        }
      }
    }
    // All refined.
    {
      std::vector<std::size_t> indices(8);
      for (std::size_t n = 0; n != indices.size(); ++n) {
        indices[n] = n;
      }
      grid.refine(indices);
      for (std::size_t n = 0; n != indices.size(); ++n) {
        grid[n][0] = n;
      }
      VoxelPatch patch;

      {
        const IndexList i = {{0, 0, 0}};
        grid.getVoxelPatch(i, &patch);
        assert(patch(0, 0, 0) == 0);
        assert(patch(1, 0, 0) == 1);
        assert(patch(0, 1, 0) == 2);
        assert(patch(1, 1, 0) == 3);
        assert(patch(0, 0, 1) == 4);
        assert(patch(1, 0, 1) == 5);
        assert(patch(0, 1, 1) == 6);
        assert(patch(1, 1, 1) == 7);
      }
      {
        const IndexList i = {{1, 0, 0}};
        grid.getVoxelPatch(i, &patch);
        assert(patch(0, 0, 0) == 1);
        assert(patch(1, 0, 0) == 1);
        assert(patch(0, 1, 0) == 3);
        assert(patch(1, 1, 0) == 1);
        assert(patch(0, 0, 1) == 5);
        assert(patch(1, 0, 1) == 1);
        assert(patch(0, 1, 1) == 7);
        assert(patch(1, 1, 1) == 1);
      }
    }
  }

  // coarsen()
  // CONTINUE HERE

  // patchDependencies() and getIntersectingPatches()
  // 1-D
  {
    const std::size_t Dimension = 1;
    typedef float Number;
    typedef levelSet::Grid<Number, Dimension, 1> Grid;
    typedef geom::BBox<Number, Dimension> BBox;
    typedef geom::Ball<Number, Dimension> Ball;

    BBox domain = {{{0}}, {{3}}};
    Grid grid(domain, Number(1.001));
    assert(grid.extents()[0] == 4);

    // patchDependencies()
    container::StaticArrayOfArrays<unsigned> dependencies;
    // Tests with a single ball.
    {
      Ball ball = {{{ -1}}, 0.25};
      patchDependencies(grid, &ball, &ball + 1, &dependencies);
      assert(dependencies.empty());
    }
    {
      Ball ball = {{{5}}, 0.25};
      patchDependencies(grid, &ball, &ball + 1, &dependencies);
      assert(dependencies.empty());
    }
    {
      Ball ball = {{{0.5}}, 0.25};
      patchDependencies(grid, &ball, &ball + 1, &dependencies);
      assert(dependencies.size() == 1);
      assert(dependencies(0, 0) == 0);
    }
    {
      Ball ball = {{{1.5}}, 0.25};
      patchDependencies(grid, &ball, &ball + 1, &dependencies);
      assert(dependencies.size() == 1);
      assert(dependencies(1, 0) == 0);
    }
    {
      Ball ball = {{{2.5}}, 0.25};
      patchDependencies(grid, &ball, &ball + 1, &dependencies);
      assert(dependencies.size() == 1);
      assert(dependencies(2, 0) == 0);
    }
    {
      Ball ball = {{{1}}, 0.25};
      patchDependencies(grid, &ball, &ball + 1, &dependencies);
      assert(dependencies.size() == 2);
      assert(dependencies(0, 0) == 0);
      assert(dependencies(1, 0) == 0);
    }
    {
      Ball ball = {{{2}}, 0.25};
      patchDependencies(grid, &ball, &ball + 1, &dependencies);
      assert(dependencies.size() == 2);
      assert(dependencies(1, 0) == 0);
      assert(dependencies(2, 0) == 0);
    }
    {
      Ball ball = {{{1.5}}, 1};
      patchDependencies(grid, &ball, &ball + 1, &dependencies);
      assert(dependencies.size() == 3);
      assert(dependencies(0, 0) == 0);
      assert(dependencies(1, 0) == 0);
      assert(dependencies(2, 0) == 0);
    }
    {
      Ball ball = {{{1.5}}, 10};
      patchDependencies(grid, &ball, &ball + 1, &dependencies);
      assert(dependencies.size() == 4);
      assert(dependencies(0, 0) == 0);
      assert(dependencies(1, 0) == 0);
      assert(dependencies(2, 0) == 0);
    }
    // Tests with multiple balls.
    {
      std::array<Ball, 2> balls = {{
          {{{ -1}}, 0.25},
          {{{5}}, 0.25}
        }
      };
      patchDependencies(grid, balls.begin(), balls.end(), &dependencies);
      assert(dependencies.empty());
    }
    {
      std::array<Ball, 2> balls = {{
          {{{0}}, 0.25},
          {{{0}}, 0.25}
        }
      };
      patchDependencies(grid, balls.begin(), balls.end(), &dependencies);
      assert(dependencies.size() == 2);
      assert(dependencies(0, 0) == 0);
      assert(dependencies(0, 1) == 1);
    }
    {
      std::array<Ball, 2> balls = {{
          {{{0}}, 0.25},
          {{{1}}, 0.25}
        }
      };
      patchDependencies(grid, balls.begin(), balls.end(), &dependencies);
      assert(dependencies.size() == 3);
      assert(dependencies(0, 0) == 0);
      assert(dependencies(0, 1) == 1);
      assert(dependencies(1, 0) == 1);
    }

    // getIntersectingPatches()
    std::vector<std::size_t> indices;
    {
      const geom::BBox<Number, Dimension> box = {{{ -2}}, {{ -1}}};
      indices.clear();
      getIntersectingPatches(grid, box, std::back_inserter(indices));
      assert(indices.empty());
    }
    {
      const geom::BBox<Number, Dimension> box = {{{4}}, {{5}}};
      indices.clear();
      getIntersectingPatches(grid, box, std::back_inserter(indices));
      assert(indices.empty());
    }
    {
      const geom::BBox<Number, Dimension> box = {{{ -2}}, {{0.1}}};
      indices.clear();
      getIntersectingPatches(grid, box, std::back_inserter(indices));
      assert(indices.size() == 1);
      assert(indices[0] == 0);
    }
    {
      const geom::BBox<Number, Dimension> box = {{{0.1}}, {{0.9}}};
      indices.clear();
      getIntersectingPatches(grid, box, std::back_inserter(indices));
      assert(indices.size() == 1);
      assert(indices[0] == 0);
    }
    {
      const geom::BBox<Number, Dimension> box = {{{0.1}}, {{1.1}}};
      indices.clear();
      getIntersectingPatches(grid, box, std::back_inserter(indices));
      assert(indices.size() == 2);
      assert(indices[0] == 0);
      assert(indices[1] == 1);
    }
    {
      const geom::BBox<Number, Dimension> box = {{{0.1}}, {{2.1}}};
      indices.clear();
      getIntersectingPatches(grid, box, std::back_inserter(indices));
      assert(indices.size() == 3);
      assert(indices[0] == 0);
      assert(indices[1] == 1);
      assert(indices[2] == 2);
    }
    {
      const geom::BBox<Number, Dimension> box = {{{1.1}}, {{2.1}}};
      indices.clear();
      getIntersectingPatches(grid, box, std::back_inserter(indices));
      assert(indices.size() == 2);
      assert(indices[0] == 1);
      assert(indices[1] == 2);
    }
    {
      const geom::BBox<Number, Dimension> box = {{{2.1}}, {{2.1}}};
      indices.clear();
      getIntersectingPatches(grid, box, std::back_inserter(indices));
      assert(indices.size() == 1);
      assert(indices[0] == 2);
    }
  }

  // patchDependencies() and getIntersectingPatches()
  // 2-D
  {
    const std::size_t Dimension = 2;
    typedef float Number;
    typedef levelSet::Grid<Number, Dimension, 1> Grid;
    typedef Grid::IndexList IndexList;
    typedef geom::BBox<Number, Dimension> BBox;
    typedef geom::Ball<Number, Dimension> Ball;

    BBox domain = {{{2, 3}}, {{4, 6}}};
    Grid grid(domain, 1.001);
    assert(grid.extents() == (IndexList{{3, 4}}));

    // patchDependencies()
    container::StaticArrayOfArrays<unsigned> dependencies;
    // Tests with a single ball.
    {
      Ball ball = {{{1, 2}}, 0.25};
      patchDependencies(grid, &ball, &ball + 1, &dependencies);
      assert(dependencies.empty());
    }
    {
      Ball ball = {{{6, 8}}, 0.25};
      patchDependencies(grid, &ball, &ball + 1, &dependencies);
      assert(dependencies.empty());
    }
    {
      Ball ball = {{{2.5, 3.5}}, 0.25};
      patchDependencies(grid, &ball, &ball + 1, &dependencies);
      assert(dependencies.size() == 1);
      assert(dependencies(0, 0) == 0);
    }
    {
      Ball ball = {{{3.5, 3.5}}, 0.25};
      patchDependencies(grid, &ball, &ball + 1, &dependencies);
      assert(dependencies.size() == 1);
      assert(dependencies(1, 0) == 0);
    }
    {
      Ball ball = {{{4.5, 3.5}}, 0.25};
      patchDependencies(grid, &ball, &ball + 1, &dependencies);
      assert(dependencies.size() == 1);
      assert(dependencies(2, 0) == 0);
    }
    {
      Ball ball = {{{3, 3.5}}, 0.25};
      patchDependencies(grid, &ball, &ball + 1, &dependencies);
      assert(dependencies.size() == 2);
      assert(dependencies(0, 0) == 0);
      assert(dependencies(1, 0) == 0);
    }
    {
      Ball ball = {{{4, 3.5}}, 0.25};
      patchDependencies(grid, &ball, &ball + 1, &dependencies);
      assert(dependencies.size() == 2);
      assert(dependencies(1, 0) == 0);
      assert(dependencies(2, 0) == 0);
    }
    {
      Ball ball = {{{2.5, 4.5}}, 0.25};
      patchDependencies(grid, &ball, &ball + 1, &dependencies);
      assert(dependencies.size() == 1);
      assert(dependencies(3, 0) == 0);
    }
    {
      Ball ball = {{{2.5, 5.5}}, 0.25};
      patchDependencies(grid, &ball, &ball + 1, &dependencies);
      assert(dependencies.size() == 1);
      assert(dependencies(6, 0) == 0);
    }
    {
      Ball ball = {{{2.5, 6.5}}, 0.25};
      patchDependencies(grid, &ball, &ball + 1, &dependencies);
      assert(dependencies.size() == 1);
      assert(dependencies(9, 0) == 0);
    }
    // Tests with multiple balls.
    {
      std::array<Ball, 2> balls = {{
          {{{1, 2}}, 0.25},
          {{{6, 8}}, 0.25}
        }
      };
      patchDependencies(grid, balls.begin(), balls.end(), &dependencies);
      assert(dependencies.empty());
    }
    {
      std::array<Ball, 2> balls = {{
          {{{2.5, 3.5}}, 0.25},
          {{{2.5, 3.5}}, 0.25}
        }
      };
      patchDependencies(grid, balls.begin(), balls.end(), &dependencies);
      assert(dependencies.size() == 2);
      assert(dependencies(0, 0) == 0);
      assert(dependencies(0, 1) == 1);
    }

    // getIntersectingPatches()
    // {{{2, 3}}, {{5, 7}}};
    std::vector<std::size_t> indices;
    {
      const geom::BBox<Number, Dimension> box = {{{0, 0}}, {{1, 1}}};
      indices.clear();
      getIntersectingPatches(grid, box, std::back_inserter(indices));
      assert(indices.empty());
    }
    {
      const geom::BBox<Number, Dimension> box = {{{8, 8}}, {{9, 9}}};
      indices.clear();
      getIntersectingPatches(grid, box, std::back_inserter(indices));
      assert(indices.empty());
    }
    {
      const geom::BBox<Number, Dimension> box = {{{2, 3}}, {{5, 7}}};
      indices.clear();
      getIntersectingPatches(grid, box, std::back_inserter(indices));
      assert(indices.size() == 12);
      for (std::size_t i = 0; i != indices.size(); ++i) {
        assert(indices[i] == i);
      }
    }
    {
      const geom::BBox<Number, Dimension> box = {{{2.1, 3.1}}, {{2.9, 3.9}}};
      indices.clear();
      getIntersectingPatches(grid, box, std::back_inserter(indices));
      assert(indices.size() == 1);
      assert(indices[0] == 0);
    }
    {
      const geom::BBox<Number, Dimension> box = {{{2.1, 3.1}}, {{3.9, 3.9}}};
      indices.clear();
      getIntersectingPatches(grid, box, std::back_inserter(indices));
      assert(indices.size() == 2);
      assert(indices[0] == 0);
      assert(indices[1] == 1);
    }
    {
      const geom::BBox<Number, Dimension> box = {{{2.1, 3.1}}, {{2.9, 4.9}}};
      indices.clear();
      getIntersectingPatches(grid, box, std::back_inserter(indices));
      assert(indices.size() == 2);
      assert(indices[0] == 0);
      assert(indices[1] == 3);
    }
    {
      const geom::BBox<Number, Dimension> box = {{{4.1, 6.1}}, {{4.9, 6.9}}};
      indices.clear();
      getIntersectingPatches(grid, box, std::back_inserter(indices));
      assert(indices.size() == 1);
      assert(indices[0] == 11);
    }
  }
  // writeVtkXml()
  {
    // 2-D
    const std::size_t Dimension = 2;
    const std::size_t N = 8;
    typedef float Number;
    typedef levelSet::Grid<Number, Dimension, N> Grid;
    typedef Grid::VertexPatch VertexPatch;
    typedef Grid::BBox BBox;
    typedef Grid::Point Point;
    typedef VertexPatch::IndexList IndexList;
    typedef geom::Ball<Number, Dimension> Ball;
    typedef container::SimpleMultiIndexRangeIterator<Dimension> Iterator;

    const BBox domain = {{{0, 0}}, {{1, 1}}};
    Grid grid(domain, 1. / (2 * N - 1) * 1.01);
    assert(grid.extents() == ext::filled_array<IndexList>(2));
    {
      std::vector<std::size_t> indices(grid.size());
      for (std::size_t i = 0; i != indices.size(); ++i) {
        indices[i] = i;
      }
      grid.refine(indices);
    }
    // Compute the distance to a ball.
    const Ball ball = {{{0.5, 0.5}}, 0.5};
    const Iterator pEnd = Iterator::end(grid.extents());
    for (Iterator p = Iterator::begin(grid.extents()); p != pEnd; ++p) {
      VertexPatch& patch = grid(*p);
      const Iterator iEnd = Iterator::end(patch.extents());
      for (Iterator i = Iterator::begin(patch.extents()); i != iEnd; ++i) {
        const Point x = grid.indexToLocation(*p, *i);
        patch(*i) = stlib::ext::euclideanDistance(ball.center, x) - ball.radius;
      }
    }
    std::ostringstream out;
    writeVtkXml(grid, out);
  }
  {
    // 3-D
    const std::size_t Dimension = 3;
    const std::size_t N = 8;
    typedef float Number;
    typedef levelSet::Grid<Number, Dimension, N> Grid;
    typedef Grid::VertexPatch VertexPatch;
    typedef Grid::BBox BBox;
    typedef Grid::Point Point;
    typedef VertexPatch::IndexList IndexList;
    typedef geom::Ball<Number, Dimension> Ball;
    typedef container::SimpleMultiIndexRangeIterator<Dimension> Iterator;

    const BBox domain = {{{0, 0, 0}}, {{1, 1, 1}}};
    Grid grid(domain, 1. / (2 * N - 1) * 1.01);
    assert(grid.extents() == ext::filled_array<IndexList>(2));
    {
      std::vector<std::size_t> indices(grid.size());
      for (std::size_t i = 0; i != indices.size(); ++i) {
        indices[i] = i;
      }
      grid.refine(indices);
    }
    // Compute the distance to a ball.
    const Ball ball = {{{0.5, 0.5, 0.5}}, 0.5};
    const Iterator pEnd = Iterator::end(grid.extents());
    for (Iterator p = Iterator::begin(grid.range()); p != pEnd; ++p) {
      VertexPatch& patch = grid(*p);
      const Iterator iEnd = Iterator::end(patch.extents());
      for (Iterator i = Iterator::begin(patch.extents()); i != iEnd; ++i) {
        const Point x = grid.indexToLocation(*p, *i);
        patch(*i) = stlib::ext::euclideanDistance(ball.center, x) - ball.radius;
      }
    }
    std::ostringstream out;
    writeVtkXml(grid, out);
  }

  return 0;
}
