// -*- C++ -*-

#include "stlib/levelSet/GridGeometry.h"

#include "stlib/numerical/equality.h"
#include "stlib/geom/kernel/Ball.h"

#include <iostream>

using namespace stlib;

template<std::size_t D>
void
test()
{
  using numerical::areEqual;

  typedef double Number;
  typedef std::array<Number, D> Point;
  typedef geom::BBox<Number, D> BBox;
  typedef levelSet::GridGeometry<D, 1, Number> GridGeometry;

  std::cout << "\nDimension = " << D << '\n';

  {
    // GridGeometry spacing.
    const Point lower = ext::filled_array<Point>(0);
    const Point upper = ext::filled_array<Point>(1);
    const BBox domain = {lower, upper};
    {
      GridGeometry x(domain, 2);
      assert(areEqual(x.spacing, 1));
    }
    {
      GridGeometry x(domain, 1.001);
      assert(areEqual(x.spacing, 1));
    }
    {
      GridGeometry x(domain, 0.5001);
      assert(areEqual(x.spacing, 0.5));
    }
    {
      GridGeometry x(domain, 0.1001);
      assert(areEqual(x.spacing, 0.1));
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

  // report()
  {
    const std::size_t Dimension = 2;
    const std::size_t N = 4;
    typedef float Number;
    typedef levelSet::GridGeometry<Dimension, N, Number> GridGeometry;
    typedef GridGeometry::IndexList IndexList;
    typedef GridGeometry::Range Range;
    typedef geom::BBox<Number, Dimension> BBox;
    typedef std::pair<IndexList, IndexList> DualIndices;

    BBox domain = {{{0, 0}}, {{1, 1}}};
    GridGeometry grid(domain, Number(1. / (8 - 1) * 1.001));
    assert(grid.gridExtents == ext::filled_array<IndexList>(2));

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
  }

  // patchDependencies()
  // 1-D
  {
    const std::size_t Dimension = 1;
    typedef float Number;
    typedef levelSet::GridGeometry<Dimension, 1, Number> GridGeometry;
    typedef geom::BBox<Number, Dimension> BBox;
    typedef geom::Ball<Number, Dimension> Ball;

    BBox domain = {{{0}}, {{3}}};
    GridGeometry grid(domain, Number(1.001));
    assert(grid.gridExtents[0] == 4);

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
    typedef levelSet::GridGeometry<Dimension, 1, Number> GridGeometry;
    typedef GridGeometry::IndexList IndexList;
    typedef geom::BBox<Number, Dimension> BBox;
    typedef geom::Ball<Number, Dimension> Ball;

    BBox domain = {{{2, 3}}, {{4, 6}}};
    GridGeometry grid(domain, 1.001);
    assert(grid.gridExtents == (IndexList{{3, 4}}));

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

  return 0;
}
