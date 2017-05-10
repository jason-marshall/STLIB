// -*- C++ -*-

#include "stlib/levelSet/components.h"

using namespace stlib;

int
main()
{
  // 2-D tests.
  {
    const std::size_t Dimension = 2;
    const std::size_t N = 8;
    typedef float Number;
    typedef levelSet::Grid<Number, Dimension, N> FunctionGrid;
    typedef levelSet::Grid<unsigned, Dimension, N, Number> ComponentGrid;
    typedef FunctionGrid::IndexList IndexList;
    typedef geom::BBox<Number, Dimension> BBox;

    // Tests with a single patch.
    {
      const BBox domain = {{{0, 0}}, {{1, 1}}};
      FunctionGrid grid(domain, 1.);
      assert(grid.extents() == ext::filled_array<IndexList>(1));
      // Unrefined.
      // Zero components.
      grid[0].fillValue = 1;
      ComponentGrid components(domain, grid.extents());
      assert(labelComponents(grid, &components) == 0);
      assert(components[0].fillValue ==
             std::numeric_limits<unsigned>::max());
      // One component.
      grid[0].fillValue = 0;
      assert(labelComponents(grid, &components) == 1);
      assert(components[0].fillValue == 0);
      // Refined.
      {
        std::vector<std::size_t> indices(1, 0);
        grid.refine(indices);
      }
      // Zero components.
      assert(labelComponents(grid, &components) == 0);
      assert(components[0].isRefined());
      for (std::size_t i = 0; i != components[0].size(); ++i) {
        assert(components[0][i] ==
               std::numeric_limits<unsigned>::max());
      }
      // One component.
      grid[0][0] = -1;
      assert(labelComponents(grid, &components) == 1);
      assert(components[0].isRefined());
      assert(components[0][0] == 0);
      for (std::size_t i = 1; i != components[0].size(); ++i) {
        assert(components[0][i] ==
               std::numeric_limits<unsigned>::max());
      }
      // Two components.
      grid[0][2] = -1;
      assert(labelComponents(grid, &components) == 2);
      assert(components[0].isRefined());
      assert(components[0][0] == 0);
      assert(components[0][1] == std::numeric_limits<unsigned>::max());
      assert(components[0][2] == 1);
      for (std::size_t i = 3; i != components[0].size(); ++i) {
        assert(components[0][i] ==
               std::numeric_limits<unsigned>::max());
      }
      // One component.
      grid[0][1] = 0;
      assert(labelComponents(grid, &components) == 1);
      assert(components[0].isRefined());
      assert(components[0][0] == 0);
      assert(components[0][1] == 0);
      assert(components[0][2] == 0);
      for (std::size_t i = 3; i != components[0].size(); ++i) {
        assert(components[0][i] ==
               std::numeric_limits<unsigned>::max());
      }
    }
    // Tests with four patches.
    {
      const BBox domain = {{{0, 0}}, {{1, 1}}};
      FunctionGrid grid(domain, 1. / (2 * N - 1) * 1.01);
      assert(grid.extents() == ext::filled_array<IndexList>(2));
      // Unrefined.
      // Zero components.
      for (std::size_t i = 0; i != grid.size(); ++i) {
        grid[i].fillValue = 1;
      }
      ComponentGrid components(domain, grid.extents());
      assert(labelComponents(grid, &components) == 0);
      for (std::size_t i = 0; i != components.size(); ++i) {
        assert(components[i].fillValue ==
               std::numeric_limits<unsigned>::max());
      }
      // One component.
      grid[0].fillValue = 0;
      assert(labelComponents(grid, &components) == 1);
      assert(components[0].fillValue == 0);
      for (std::size_t i = 1; i != components.size(); ++i) {
        assert(components[i].fillValue ==
               std::numeric_limits<unsigned>::max());
      }
      // Refined.
      {
        std::vector<std::size_t> indices(1, 0);
        grid.refine(indices);
      }
      // Zero components.
      assert(labelComponents(grid, &components) == 0);
      assert(components[0].isRefined());
      for (std::size_t i = 0; i != components[0].size(); ++i) {
        assert(components[0][i] ==
               std::numeric_limits<unsigned>::max());
      }
      for (std::size_t i = 1; i != components.size(); ++i) {
        assert(components[i].fillValue ==
               std::numeric_limits<unsigned>::max());
      }
      // One component.
      grid[0][0] = -1;
      assert(labelComponents(grid, &components) == 1);
      assert(components[0].isRefined());
      assert(components[0][0] == 0);
      for (std::size_t i = 1; i != components[0].size(); ++i) {
        assert(components[0][i] ==
               std::numeric_limits<unsigned>::max());
      }
      for (std::size_t i = 1; i != components.size(); ++i) {
        assert(components[i].fillValue ==
               std::numeric_limits<unsigned>::max());
      }
      grid[0][0] = 1;
      // One component.
      grid[1].fillValue = -1;
      assert(labelComponents(grid, &components) == 1);
      assert(components[0].isRefined());
      for (std::size_t i = 0; i != components[0].size(); ++i) {
        assert(components[0][i] ==
               std::numeric_limits<unsigned>::max());
      }
      assert(components[1].fillValue == 0);
      for (std::size_t i = 2; i != components.size(); ++i) {
        assert(components[i].fillValue ==
               std::numeric_limits<unsigned>::max());
      }
    }
  }

  return 0;
}
