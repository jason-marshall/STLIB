// -*- C++ -*-

#include "stlib/numerical/interpolation/PolynomialInterpolationNonNegative.h"
#include "stlib/numerical/equality.h"

#include <iostream>

using namespace stlib;

int
main()
{
  using numerical::areEqual;
  const double Eps = std::numeric_limits<double>::epsilon();

  // Linear
  {
    typedef numerical::PolynomialInterpolationNonNegative<double, 1> F;

    {
      F f;
      assert(f.getNumberOfCells() == 0);
      assert(f.getNumberOfGridPoints() == 1);
    }
    {
      F f(2, 0., 1.);
      assert(f.getNumberOfCells() == 1);
      assert(f.getNumberOfGridPoints() == 2);
    }
    {
      double grid[] = {1, 2};
      const std::size_t Size = sizeof(grid) / sizeof(double);
      const double Lower = 0;
      const double Upper = 1;
      F f(grid, Size, Lower, Upper);
      assert(f.getNumberOfCells() == 1);
      assert(f.getNumberOfGridPoints() == 2);
      {
        assert(areEqual(f(0), 1));
        assert(std::abs(f(0.5) - 1.5) < 0.1);
        assert(areEqual(f(1 - Eps), 2));
      }
    }
    {
      double grid[] = {0, 1};
      const std::size_t Size = sizeof(grid) / sizeof(double);
      const double Lower = 0;
      const double Upper = 1;
      F f(grid, Size, Lower, Upper);
      assert(f.getNumberOfCells() == 1);
      assert(f.getNumberOfGridPoints() == 2);
      {
        assert(areEqual(f(0), 0));
        assert(std::abs(f(0.5) - 0.5) < 0.1);
      }
    }
    {
      double grid[] = {2, 3, 5, 7};
      const std::size_t Size = sizeof(grid) / sizeof(double);
      const double Lower = -1;
      const double Upper = 2;
      F f(grid, Size, Lower, Upper);
      {
        assert(areEqual(f(-1), grid[0]));
        assert(areEqual(f(0), grid[1]));
        assert(areEqual(f(1), grid[2]));
        assert(areEqual(f(2 - 5 * Eps), grid[3], 10));
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
      f.setGridValues(grid);
      {
        assert(areEqual(f(-1), grid[0]));
        assert(areEqual(f(0), grid[1]));
        assert(areEqual(f(1), grid[2]));
        assert(areEqual(f(2 - 5 * Eps), grid[3], 10));
      }
    }
  }

  // Cubic
  {
    typedef numerical::PolynomialInterpolationNonNegative<double, 3> F;
    {
      // x
      double grid[] = {0, 1};
      const std::size_t Size = sizeof(grid) / sizeof(double);
      const double Lower = 0;
      const double Upper = 1;
      F f(grid, Size, Lower, Upper);
      {
        assert(areEqual(f(0), 0));
        assert(std::abs(f(0.5) - 0.5) < 0.1);
      }
    }
  }

  // Quintic
  {
    typedef numerical::PolynomialInterpolationNonNegative<double, 5> F;
    {
      // x
      double grid[] = {0, 1};
      const std::size_t Size = sizeof(grid) / sizeof(double);
      const double Lower = 0;
      const double Upper = 1;
      F f(grid, Size, Lower, Upper);
      {
        assert(areEqual(f(0), 0));
        assert(std::abs(f(0.5) - 0.5) < 0.1);
      }
    }
    {
      // x
      double grid[] = {0, 1};
      const std::size_t Size = sizeof(grid) / sizeof(double);
      const double Lower = 0;
      const double Upper = 2;
      F f(grid, Size, Lower, Upper);
      {
        assert(areEqual(f(0), 0));
        assert(std::abs(f(0.5) - 0.25) < 0.1);
      }
    }
  }

  //
  // Example code.
  //

  // Linear.
  {
    typedef numerical::PolynomialInterpolationNonNegative<double, 1> F;
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
    F f(grid.begin(), grid.size(), Lower, Upper);
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
    f.setGridValues(grid.begin());
  }

  // Cubic.
  {
    typedef numerical::PolynomialInterpolationNonNegative<double, 3> F;
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
    F f(grid.begin(), grid.size(), Lower, Upper);
    // Check that the function has reasonable values at the cell centers.
    const std::size_t numberOfCells = grid.size() - 1;
    for (std::size_t i = 0; i != numberOfCells; ++i) {
      const double x = Dx * (i + 0.5);
      assert(std::abs(f(x) - std::exp(x)) < Dx * Dx);
    }
  }
  // Quintic.
  {
    typedef numerical::PolynomialInterpolationNonNegative<double, 5> F;
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
    F f(grid.begin(), grid.size(), Lower, Upper);
    // Check that the function has reasonable values at the cell centers.
    const std::size_t numberOfCells = grid.size() - 1;
    for (std::size_t i = 0; i != numberOfCells; ++i) {
      const double x = Dx * (i + 0.5);
      // Check the function value.
      assert(std::abs(f(x) - std::exp(x)) < Dx * Dx);
    }
  }

  //
  // Troubling case.
  //
  {
    std::vector<double> grid(6);
    const double Lower = 0;
    const double Upper = 0.25;
    grid[0] = 1.47E+29;
    grid[1] = 9.43E+12;
    grid[2] = 1.47E+11;
    grid[3] = 1.29E+10;
    grid[4] = 2.30E+09;
    grid[5] = 6.03E+08;
    {
      std::cout << "Quintic.\n";
      typedef numerical::PolynomialInterpolationNonNegative<double, 5>
      F;
      F f(grid.begin(), grid.size(), Lower, Upper);
      for (double x = 0; x < 0.25; x += 0.01) {
        std::cout << x << ' ' << f(x) << '\n';
      }
    }
    {
      std::cout << "\nCubic.\n";
      typedef numerical::PolynomialInterpolationNonNegative<double, 3>
      F;
      F f(grid.begin(), grid.size(), Lower, Upper);
      for (double x = 0; x < 0.25; x += 0.01) {
        std::cout << x << ' ' << f(x) << '\n';
      }
    }
    {
      std::cout << "\nLinear.\n";
      typedef numerical::PolynomialInterpolationNonNegative<double, 1>
      F;
      F f(grid.begin(), grid.size(), Lower, Upper);
      for (double x = 0; x < 0.25; x += 0.01) {
        std::cout << x << ' ' << f(x) << '\n';
      }
    }
  }

  return 0;
}
