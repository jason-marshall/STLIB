// -*- C++ -*-

#include "stlib/levelSet/marchingSimplices.h"
#include "stlib/levelSet/positiveDistance.h"
#include "stlib/levelSet/negativeDistance.h"
#include "stlib/levelSet/countGrid.h"
#include "stlib/levelSet/flood.h"

#include "stlib/numerical/constants.h"
#include "stlib/numerical/equality.h"

using namespace stlib;

int
main()
{
  using numerical::areEqual;
  using levelSet::content;
  using levelSet::voxelContent;

  typedef float T;
  // 1-D
  {
    const std::size_t D = 1;

    // Voxel.
    // Note that the values must have mixed signs.
    {
      const std::array<T, 2> data = {{ -1, 1}};
      const container::EquilateralArray<T, D, 2> voxel(data);
      const T spacing = 1;
      assert(areEqual(voxelContent(voxel, spacing), T(0.5)));
    }
    {
      const std::array<T, 2> data = {{1, -1}};
      const container::EquilateralArray<T, D, 2> voxel(data);
      const T spacing = 1;
      assert(areEqual(voxelContent(voxel, spacing), T(0.5)));
    }
    {
      const std::array<T, 2> data = {{ -2, 2}};
      const container::EquilateralArray<T, D, 2> voxel(data);
      const T spacing = 1;
      assert(areEqual(voxelContent(voxel, spacing), T(0.5)));
    }
    {
      const std::array<T, 2> data = {{ -1, 1}};
      const container::EquilateralArray<T, D, 2> voxel(data);
      const T spacing = 2;
      assert(areEqual(voxelContent(voxel, spacing), T(1)));
    }
    // Patch.
    {
      const std::array<T, 4> data = {{ -2, -1, 1, 2}};
      const container::EquilateralArray<T, D, 4> patch(data);
      const T spacing = 1;
      assert(areEqual(content(patch, spacing), T(1.5)));
    }
    {
      const std::array<T, 4> data = {{ -2, -1, -1, -2}};
      const container::EquilateralArray<T, D, 4> patch(data);
      const T spacing = 1;
      assert(areEqual(content(patch, spacing), T(3)));
    }
    {
      const std::array<T, 4> data = {{2, 1, 1, 2}};
      const container::EquilateralArray<T, D, 4> patch(data);
      const T spacing = 1;
      assert(areEqual(content(patch, spacing), T(0)));
    }
    {
      const std::array<T, 4> data = {{ -2, -1, 0, 2}};
      const container::EquilateralArray<T, D, 4> patch(data);
      const T spacing = 1;
      assert(areEqual(content(patch, spacing), T(2)));
    }
    // Grid.
    {
      const std::size_t N = 2;
      typedef levelSet::Grid<T, D, N> Grid;
      typedef Grid::IndexList IndexList;
      typedef Grid::BBox BBox;
      typedef geom::Ball<T, D> Ball;

      BBox domain = {{{ -1}}, {{1}}};
      Grid grid(domain, T(2. / (3 * N - 1) * 1.001));
      assert(grid.extents() == (IndexList{{3}}));

      std::vector<Ball> balls;
      balls.push_back(Ball{{{0}}, T(0.5)});
      levelSet::positiveDistance(&grid, balls, T(0), grid.spacing);
      levelSet::floodFill(&grid);
      assert(! hasUnknown(grid));
      assert(areEqual(content(grid), T(1)));
      // Use components.
      std::vector<T> content, boundary;
      contentAndBoundary(grid, &content, &boundary);
      assert(content.size() == 1);
      assert(boundary.size() == 1);
      assert(areEqual(content[0], T(1)));
    }
  }

  // 2-D
  {
    const std::size_t D = 2;

    //
    // Voxel.
    //
    // Note that the values must have mixed signs.
    {
      //  0  1
      // -1  0
      const std::array<T, 4> data = {{ -1, 0, 0, 1}};
      const container::EquilateralArray<T, D, 2> voxel(data);
      const T spacing = 1;
      assert(areEqual(voxelContent(voxel, spacing), T(0.5)));
    }
    {
      //  1  0
      //  0 -1
      const std::array<T, 4> data = {{0, -1, 1, 0}};
      const container::EquilateralArray<T, D, 2> voxel(data);
      const T spacing = 1;
      assert(areEqual(voxelContent(voxel, spacing), T(0.5)));
    }
    {
      //  0 -1
      //  1  0
      const std::array<T, 4> data = {{1, 0, 0, -1}};
      const container::EquilateralArray<T, D, 2> voxel(data);
      const T spacing = 1;
      assert(areEqual(voxelContent(voxel, spacing), T(0.5)));
    }
    {
      // -1  0
      //  0  1
      const std::array<T, 4> data = {{0, 1, -1, 0}};
      const container::EquilateralArray<T, D, 2> voxel(data);
      const T spacing = 1;
      assert(areEqual(voxelContent(voxel, spacing), T(0.5)));
    }
    {
      const std::array<T, 4> data = {{ -2, 0, 0, 2}};
      const container::EquilateralArray<T, D, 2> voxel(data);
      const T spacing = 1;
      assert(areEqual(voxelContent(voxel, spacing), T(0.5)));
    }
    {
      const std::array<T, 4> data = {{ -1, 0, 0, 1}};
      const container::EquilateralArray<T, D, 2> voxel(data);
      const T spacing = 2;
      assert(areEqual(voxelContent(voxel, spacing), T(2)));
    }
    {
      const std::array<T, 4> data = {{0, -1, 2, 0}};
      const container::EquilateralArray<T, D, 2> voxel(data);
      const T spacing = 1;
      assert(areEqual(voxelContent(voxel, spacing), T(0.5)));
    }
    {
      const std::array<T, 4> data = {{1, -1, 2, 0}};
      const container::EquilateralArray<T, D, 2> voxel(data);
      const T spacing = 1;
      assert(areEqual(voxelContent(voxel, spacing), T(0.25)));
    }
    {
      const std::array<T, 4> data = {{0, -1, 2, 1}};
      const container::EquilateralArray<T, D, 2> voxel(data);
      const T spacing = 1;
      assert(areEqual(voxelContent(voxel, spacing), T(0.25)));
    }
    {
      const std::array<T, 4> data = {{1, -1, 2, 1}};
      const container::EquilateralArray<T, D, 2> voxel(data);
      const T spacing = 1;
      assert(areEqual(voxelContent(voxel, spacing), T(0.125)));
    }
    //
    // Patch.
    //
    {
      //  1  2  3
      //  0  1  2
      // -1  0  1
      const std::array<T, 9> data = {{
          -1, 0, 1,
          0, 1, 2,
          1, 2, 3
        }
      };
      const container::EquilateralArray<T, D, 3> patch(data);
      const T spacing = 1;
      assert(areEqual(content(patch, spacing), T(0.5)));
    }
    {
      //  1  2  3
      //  0  1  2
      //  0  0  1
      const std::array<T, 9> data = {{
          0, 0, 1,
          0, 1, 2,
          1, 2, 3
        }
      };
      const container::EquilateralArray<T, D, 3> patch(data);
      const T spacing = 1;
      assert(areEqual(content(patch, spacing), T(0)));
    }
    {
      //  0  1  2
      // -1  0  1
      // -2 -1  0
      const std::array<T, 9> data = {{
          -2, -1, 0,
          -1, 0, 1,
          0, 1, 2
        }
      };
      const container::EquilateralArray<T, D, 3> patch(data);
      const T spacing = 1;
      assert(areEqual(content(patch, spacing), T(2)));
    }
    {
      // -1  0  1
      // -1  0  1
      // -1  0  1
      const std::array<T, 9> data = {{
          -1, 0, 1,
          -1, 0, 1,
          -1, 0, 1,
        }
      };
      const container::EquilateralArray<T, D, 3> patch(data);
      const T spacing = 1;
      assert(areEqual(content(patch, spacing), T(2)));
    }
    {
      // -1 -1 -1
      // -1 -1 -1
      // -1 -1 -1
      const std::array<T, 9> data = {{
          -1, -1, -1,
          -1, -1, -1,
          -1, -1, -1,
        }
      };
      const container::EquilateralArray<T, D, 3> patch(data);
      const T spacing = 1;
      assert(areEqual(content(patch, spacing), T(4)));
    }
    //
    // Grid.
    //
    {
      const std::size_t N = 2;
      typedef levelSet::Grid<T, D, N> Grid;
      typedef Grid::IndexList IndexList;
      typedef Grid::BBox BBox;
      typedef geom::Ball<T, D> Ball;

      BBox domain = {{{ -2, -2}}, {{2, 2}}};
      std::vector<Ball> balls;
      balls.push_back(Ball{{{0, 0}}, T(1)});
      const T area = numerical::Constants<T>::Pi();
      const T length = 2 * numerical::Constants<T>::Pi();
      // error = dx * dx2 * length / dx
      {
        Grid grid(domain, T(4. / (10 * N - 1) * 1.001));
        assert(grid.extents() == (IndexList{{10, 10}}));

        levelSet::positiveDistance(&grid, balls, T(0),
                                   T(std::sqrt(2.) * grid.spacing));
        levelSet::floodFill(&grid);
        assert(! hasUnknown(grid));
        const T c = content(grid);
        assert(std::abs(c - area) < grid.spacing * grid.spacing * length);
        std::cout << "Content = " << c
                  << ", error = " << std::abs(c - area)
                  << ", spacing = " << grid.spacing << '\n';
        // Use components.
        std::vector<T> content, boundary;
        contentAndBoundary(grid, &content, &boundary);
        assert(content.size() == 1);
        assert(boundary.size() == 1);
        assert(std::abs(content[0] - area) <
               grid.spacing * grid.spacing * length);
      }
      {
        Grid grid(domain, T(4. / (100 * N - 1) * 1.001));
        assert(grid.extents() == (IndexList{{100, 100}}));

        levelSet::positiveDistance(&grid, balls, T(0),
                                   T(std::sqrt(2.) * grid.spacing));
        levelSet::floodFill(&grid);
        assert(! hasUnknown(grid));
        const T c = content(grid);
        assert(std::abs(c - area) <
               1.5 * grid.spacing * grid.spacing * length);
        std::cout << "Content = " << c
                  << ", error = " << std::abs(c - area)
                  << ", spacing = " << grid.spacing << '\n';
        // Use components.
        std::vector<T> content, boundary;
        contentAndBoundary(grid, &content, &boundary);
        assert(content.size() == 1);
        assert(boundary.size() == 1);
        assert(std::abs(content[0] - area) <
               1.5 * grid.spacing * grid.spacing * length);
      }
    }
  }

  // 3-D
  {
    const std::size_t D = 3;
    typedef std::array<T, D> Point;
    const T Eps = std::numeric_limits<T>::epsilon();

    //
    // Simplex.
    //
    // Note that the values must have mixed signs.
    {
      //  0  E
      // -1  0
      const std::array<T, 4> values = {{ -1, 0, 0, Eps}};
      const std::array<Point, 4> locations = {{
          {{0, 0, 0}},
          {{1, 0, 0}},
          {{0, 1, 0}},
          {{0, 0, 1}}
        }
      };
      const T simplexContent = 1. / 6;
      assert(areEqual(content(values, locations, simplexContent), T(1. / 6)));
    }
    {
      //  0  1
      // -1  0
      const std::array<T, 4> values = {{ -1, 0, 0, 1}};
      const std::array<Point, 4> locations = {{
          {{0, 0, 0}},
          {{1, 0, 0}},
          {{0, 1, 0}},
          {{0, 0, 1}}
        }
      };
      const T simplexContent = 1. / 6;
      assert(areEqual(content(values, locations, simplexContent), T(1. / 12)));
    }
    {
      //  1  1
      // -1 -1
      const std::array<T, 4> values = {{ -1, -1, 1, 1}};
      const std::array<Point, 4> locations = {{
          {{0, 0, 0}},
          {{1, 0, 0}},
          {{0, 1, 0}},
          {{0, 0, 1}}
        }
      };
      const T simplexContent = 1. / 6;
      assert(areEqual(content(values, locations, simplexContent), T(1. / 12)));
    }
    {
      // -1  E
      // -1 -1
      const std::array<T, 4> values = {{ -1, -1, -1, Eps}};
      const std::array<Point, 4> locations = {{
          {{0, 0, 0}},
          {{1, 0, 0}},
          {{0, 1, 0}},
          {{0, 0, 1}}
        }
      };
      const T simplexContent = 1. / 6;
      assert(areEqual(content(values, locations, simplexContent), T(1. / 6)));
    }
    {
      // -1  1
      // -1 -1
      const std::array<T, 4> values = {{ -1, -1, -1, 1}};
      const std::array<Point, 4> locations = {{
          {{0, 0, 0}},
          {{1, 0, 0}},
          {{0, 1, 0}},
          {{0, 0, 1}}
        }
      };
      const T simplexContent = 1. / 6;
      assert(areEqual(content(values, locations, simplexContent),
                      T(1. / 6 - 1. / 48)));
    }
    //
    // Voxel.
    //
    // Note that the values must have mixed signs.
    {
      //  0  0    0 E
      // -1  0    0 0
      const std::array<T, 8> data = {{ -1, 0, 0, 0, 0, 0, 0, Eps}};
      const container::EquilateralArray<T, D, 2> voxel(data);
      const T spacing = 1;
      assert(areEqual(voxelContent(voxel, spacing), T(1)));
    }
    {
      //  0  0    0 1
      // -E  0    0 0
      const std::array<T, 8> data = {{ -Eps, 0, 0, 0, 0, 0, 0, 1}};
      const container::EquilateralArray<T, D, 2> voxel(data);
      const T spacing = 1;
      assert(areEqual(voxelContent(voxel, spacing), T(0)));
    }
    {
      // -1 -1    1 1
      // -1 -1    1 1
      const std::array<T, 8> data = {{ -1, -1, -1, -1, 1, 1, 1, 1}};
      const container::EquilateralArray<T, D, 2> voxel(data);
      const T spacing = 1;
      assert(areEqual(voxelContent(voxel, spacing), T(0.5)));
    }
    {
      // -2 -2    1 1
      // -2 -2    1 1
      const std::array<T, 8> data = {{ -2, -2, -2, -2, 1, 1, 1, 1}};
      const container::EquilateralArray<T, D, 2> voxel(data);
      const T spacing = 1;
      assert(areEqual(voxelContent(voxel, spacing), T(2. / 3)));
    }
    //
    // Patch.
    //
    {
      // -1 -1 -1    0  0  0    1  1  1
      // -1 -1 -1    0  0  0    1  1  1
      // -1 -1 -1    0  0  0    1  1  1
      const std::array<T, 27> data = {{
          -1, -1, -1,
          -1, -1, -1,
          -1, -1, -1,
          0, 0, 0,
          0, 0, 0,
          0, 0, 0,
          1, 1, 1,
          1, 1, 1,
          1, 1, 1
        }
      };
      const container::EquilateralArray<T, D, 3> patch(data);
      const T spacing = 1;
      assert(areEqual(content(patch, spacing), T(4)));
    }
    //
    // Grid.
    //
    {
      const std::size_t N = 2;
      typedef levelSet::Grid<T, D, N> Grid;
      typedef Grid::IndexList IndexList;
      typedef Grid::BBox BBox;
      typedef geom::Ball<T, D> Ball;

      BBox domain = {{{ -2, -2, -2}}, {{2, 2, 2}}};
      std::vector<Ball> balls;
      balls.push_back(Ball{{{0, 0, 0}}, T(1)});
      const T volume = 4 * numerical::Constants<T>::Pi() / 3;
      const T area = 4 * numerical::Constants<T>::Pi();
      // error = dx^2 * dx^2 * area / dx^2
      {
        Grid grid(domain, T(4. / (10 * N - 1) * 1.001));
        assert(grid.extents() == (IndexList{{10, 10, 10}}));

        levelSet::positiveDistance(&grid, balls, T(0),
                                   T(std::sqrt(3.) * grid.spacing));
        levelSet::floodFill(&grid);
        assert(! hasUnknown(grid));
        const T c = content(grid);
        assert(std::abs(c - volume) < grid.spacing * grid.spacing * area);
        std::cout << "Content = " << c
                  << ", error = " << std::abs(c - volume)
                  << ", spacing = " << grid.spacing << '\n';
      }
      {
        Grid grid(domain, T(4. / (50 * N - 1) * 1.001));
        assert(grid.extents() ==
               (IndexList{{50, 50, 50}}));

        levelSet::positiveDistance(&grid, balls, T(0),
                                   T(std::sqrt(3.) * grid.spacing));
        levelSet::floodFill(&grid);
        assert(! hasUnknown(grid));
        const T c = content(grid);
        assert(std::abs(c - volume) < grid.spacing * grid.spacing * area);
        std::cout << "Content = " << c
                  << ", error = " << std::abs(c - volume)
                  << ", spacing = " << grid.spacing << '\n';
      }
    }
  }

  return 0;
}
