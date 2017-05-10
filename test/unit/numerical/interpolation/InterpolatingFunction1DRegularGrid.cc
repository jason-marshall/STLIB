// -*- C++ -*-

#include "stlib/numerical/interpolation/InterpolatingFunction1DRegularGrid.h"
#include "stlib/numerical/equality.h"

using namespace stlib;

int
main()
{
  using numerical::areEqual;
  const double Eps = std::numeric_limits<double>::epsilon();

  {
    typedef numerical::InterpolatingFunction1DRegularGrid<double, 1> F;

    {
      double grid[] = {0, 1};
      const std::size_t Size = sizeof(grid) / sizeof(double);
      const double Lower = 0;
      const double Upper = 1;
      F f(grid, grid + Size, Lower, Upper);
      {
        assert(areEqual(f(0), 0));
        assert(areEqual(f(0.5), 0.5));
      }
    }
    {
      double grid[] = {2, 3, 5, 7};
      const std::size_t Size = sizeof(grid) / sizeof(double);
      const double Lower = -1;
      const double Upper = 2;
      F f(grid, grid + Size, Lower, Upper);
      {
        assert(areEqual(f(-1), grid[0]));
        assert(areEqual(f(0), grid[1]));
        assert(areEqual(f(1), grid[2]));
        assert(areEqual(f(2 - 5 * Eps), grid[3], 10));

        assert(areEqual(f(-0.9), 0.9 * grid[0] + 0.1 * grid[1]));
        assert(areEqual(f(-0.5), 0.5 * grid[0] + 0.5 * grid[1]));
        assert(areEqual(f(-0.1), 0.1 * grid[0] + 0.9 * grid[1]));
      }
      {
        // Copy constructor.
        F g = f;
        assert(f(1) == g(1));
      }
      {
        // Assignment operator.
        F g;
        g = f;
        assert(f(1) == g(1));
      }

      // Change the grid values.
      for (std::size_t i = 0; i != Size; ++i) {
        grid[i] = i * i;
      }
      f.setGridValues(grid, grid + Size);
      {
        assert(areEqual(f(-1), grid[0]));
        assert(areEqual(f(0), grid[1]));
        assert(areEqual(f(1), grid[2]));
        assert(areEqual(f(2 - 5 * Eps), grid[3], 10));

        assert(areEqual(f(-0.9), 0.9 * grid[0] + 0.1 * grid[1]));
        assert(areEqual(f(-0.5), 0.5 * grid[0] + 0.5 * grid[1]));
        assert(areEqual(f(-0.1), 0.1 * grid[0] + 0.9 * grid[1]));
      }
    }
  }

  {
    typedef numerical::InterpolatingFunction1DRegularGrid<double, 3> F;
    {
      // x
      double grid[] = {0, 1};
      const std::size_t Size = sizeof(grid) / sizeof(double);
      const double Lower = 0;
      const double Upper = 1;
      F f(grid, grid + Size, Lower, Upper);
      {
        assert(areEqual(f(0), 0));
        assert(areEqual(f(0.5), 0.5));
        double derivative;
        assert(areEqual(f(0.5, &derivative), 0.5));
        assert(areEqual(derivative, 1.));
      }
    }
    {
      // x^3.
      double grid[] = { -1, 0, 1, 8};
      const std::size_t Size = sizeof(grid) / sizeof(double);
      const double Lower = -1;
      const double Upper = 2;
      F f(grid, grid + Size, Lower, Upper);
      {
        assert(areEqual(f(-1), grid[0]));
        assert(areEqual(f(0), grid[1]));
        assert(areEqual(f(1), grid[2]));
        assert(areEqual(f(2 - 5 * Eps), grid[3], 10));
        // In[1]:= f[x_] := a + b x + c x^2 + d x^3
        // In[4]:= f[x] /. Solve[{f[0] == 0, f[1] == 1, f'[0] == 1, f'[1] == 4}][[1]]
        // Out[4]= x - 3 x^2 + 3 x^3
        // In[5]:= D[%, x]

        // Out[5]= 1 - 6 x + 9 x^2
        const double x = 0.5;
        const double x2 = x * x;
        const double x3 = x2 * x;
        assert(areEqual(f(x), x - 3 * x2 + 3 * x3));
        double derivative;
        assert(areEqual(f(x, &derivative), x - 3 * x2 + 3 * x3));
        assert(areEqual(derivative, 1 - 6 * x + 9 * x2));
      }
      {
        // Copy constructor.
        F g = f;
        assert(f(1) == g(1));
      }
      {
        // Assignment operator.
        F g;
        g = f;
        assert(f(1) == g(1));
      }

      // Change the grid values to x^2.
      // 1, 0, 1, 4
      for (std::size_t i = 0; i != Size; ++i) {
        grid[i] = (i - 1) * (i - 1);
      }
      f.setGridValues(grid, grid + Size);
      {
        assert(areEqual(f(-1), grid[0]));
        assert(areEqual(f(0), grid[1]));
        assert(areEqual(f(1), grid[2]));
        assert(areEqual(f(2 - 5 * Eps), grid[3], 10));


        // In[6]:= f[x] /. Solve[{f[0] == 0, f[1] == 1, f'[0] == 0, f'[1] == 2}][[1]]
        // Out[6]= x^2
        // In[7]:= D[%, x]
        // Out[7]= 2 x
        const double x = 0.5;
        const double x2 = x * x;
        assert(areEqual(f(x), x2));
        double derivative;
        assert(areEqual(f(x, &derivative), x2));
        assert(areEqual(derivative, 2 * x));
      }
    }
  }

  //
  // Example code.
  //

  // Linear.
  {
    typedef numerical::InterpolatingFunction1DRegularGrid<double, 1> F;
    // Make a grid to sample the exponential function on the domain [0..1).
    // Use a grid spacing of 0.1.
    std::vector<double> grid(11);
    const double Lower = 0;
    const double Upper = 1;
    const double Dx = (Upper - Lower) / (grid.size() - 1);
    for (std::size_t i = 0; i != grid.size(); ++i) {
      grid[i] = std::exp(Dx * i);
    }
    // Construct the interpolating function.
    F f(grid.begin(), grid.end(), Lower, Upper);
    // Check that the function has the correct values at the grid points.
    // Note that we cannot evaluate the interpolating function at values
    // greater than or equal to the last grid point.
    const std::size_t numberOfCells = grid.size() - 1;
    for (std::size_t i = 0; i != numberOfCells; ++i) {
      assert(numerical::areEqual(f(Dx * i), grid[i]));
    }
    // Change the interpolating function to sample the function f(x) = x^2.
    for (std::size_t i = 0; i != grid.size(); ++i) {
      grid[i] = (Dx * i) * (Dx * i);
    }
    f.setGridValues(grid.begin(), grid.end());
  }

  // Cubic.
  {
    typedef numerical::InterpolatingFunction1DRegularGrid<double, 3> F;
    // Make a grid to sample the exponential function on the domain [0..1).
    // Use a grid spacing of 0.1.
    std::vector<double> grid(11);
    const double Lower = 0;
    const double Upper = 1;
    const double Dx = (Upper - Lower) / (grid.size() - 1);
    for (std::size_t i = 0; i != grid.size(); ++i) {
      grid[i] = std::exp(Dx * i);
    }
    // Construct the interpolating function.
    F f(grid.begin(), grid.end(), Lower, Upper);
    // Check that the function has reasonable values at the cell centers.
    const std::size_t numberOfCells = grid.size() - 1;
    double derivative;
    for (std::size_t i = 0; i != numberOfCells; ++i) {
      const double x = Dx * (i + 0.5);
      assert(std::abs(f(x) - std::exp(x)) < Dx * Dx);
      // Check the derivative as well.
      assert(std::abs(f(x, &derivative) - std::exp(x)) < Dx * Dx);
      assert(std::abs(derivative - std::exp(x)) < Dx);
    }
  }

  return 0;
}
