// -*- C++ -*-

#include "stlib/numerical/interpolation/PolynomialInterpolationUsingCoefficients.h"
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
    typedef numerical::PolynomialInterpolationUsingCoefficients<double, 1> F;

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
      double grid[] = {0, 1};
      const std::size_t Size = sizeof(grid) / sizeof(double);
      const double Lower = 0;
      const double Upper = 1;
      F f(grid, Size, Lower, Upper);
      assert(f.getNumberOfCells() == 1);
      assert(f.getNumberOfGridPoints() == 2);
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
      F f(grid, Size, Lower, Upper);
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

      // Assignment operators.
      {
        F g = f;
        g += 1.;
        assert(areEqual(f(Lower) + 1., g(Lower)));
      }
      {
        F g = f;
        g *= 2.;
        assert(areEqual(f(Lower) * 2., g(Lower)));
      }
      {
        F g = f;
        g += f;
        assert(areEqual(f(Lower) + f(Lower), g(Lower)));
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

        assert(areEqual(f(-0.9), 0.9 * grid[0] + 0.1 * grid[1]));
        assert(areEqual(f(-0.5), 0.5 * grid[0] + 0.5 * grid[1]));
        assert(areEqual(f(-0.1), 0.1 * grid[0] + 0.9 * grid[1]));
      }
    }
  }

  // Cubic
  {
    typedef numerical::PolynomialInterpolationUsingCoefficients<double, 3> F;
    {
      // x
      double grid[] = {0, 1};
      const std::size_t Size = sizeof(grid) / sizeof(double);
      const double Lower = 0;
      const double Upper = 1;
      F f(grid, Size, Lower, Upper);
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
      double g0[] = { -1, 0, 1, 8};
      const std::size_t Size = sizeof(g0) / sizeof(double);
      // 3 x^2.
      double g1[] = {3, 0, 3, 12};
      const double Lower = -1;
      const double Upper = 2;
      F f(g0, g1, Size, Lower, Upper);
      {
        assert(areEqual(f(-1), g0[0]));
        assert(areEqual(f(0), g0[1]));
        assert(areEqual(f(1), g0[2]));
        assert(areEqual(f(2 - 5 * Eps), g0[3], 10));
        const double x = 0.5;
        const double x2 = x * x;
        const double x3 = x2 * x;
        assert(areEqual(f(x), x3));
        double derivative;
        assert(areEqual(f(x, &derivative), x3));
        assert(areEqual(derivative, 3 * x2));
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
        g0[i] = (i - 1.) * (i - 1.);
        g1[i] = 2. * (i - 1.);
      }
      f.setGridValues(g0, g1);
      {
        assert(areEqual(f(-1), g0[0]));
        assert(areEqual(f(0), g0[1]));
        assert(areEqual(f(1), g0[2]));
        assert(areEqual(f(2 - 5 * Eps), g0[3], 10));

        const double x = 0.5;
        const double x2 = x * x;
        assert(areEqual(f(x), x2));
        double derivative;
        assert(areEqual(f(x, &derivative), x2));
        assert(areEqual(derivative, 2 * x));
      }
    }
  }

  // Quintic
  {
    typedef numerical::PolynomialInterpolationUsingCoefficients<double, 5> F;
    {
      // x
      double grid[] = {0, 1};
      const std::size_t Size = sizeof(grid) / sizeof(double);
      const double Lower = 0;
      const double Upper = 1;
      F f(grid, Size, Lower, Upper);
      {
        assert(areEqual(f(0), 0));
        assert(areEqual(f(0.5), 0.5));
        double first;
        assert(areEqual(f(0.5, &first), 0.5));
        assert(areEqual(first, 1.));
        double second;
        assert(areEqual(f(0.5, &first, &second), 0.5));
        assert(areEqual(first, 1.));
        assert(areEqual(second, 0.));
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
        assert(areEqual(f(0.5), 0.25));
        double first;
        assert(areEqual(f(0.5, &first), 0.25));
        assert(areEqual(first, 0.5));
        double second;
        assert(areEqual(f(0.5, &first, &second), 0.25));
        assert(areEqual(first, 0.5));
        assert(areEqual(second, 0.));
      }
    }
    {
      // x^3.
      double g0[] = { -1, 0, 1, 8};
      const std::size_t Size = sizeof(g0) / sizeof(double);
      // 3 x^2.
      double g1[] = {3, 0, 3, 12};
      // 6 x.
      double g2[] = { -6, 0, 6, 12};
      const double Lower = -1;
      const double Upper = 2;
      F f(g0, g1, g2, Size, Lower, Upper);
      {
        assert(areEqual(f(-1), g0[0]));
        assert(areEqual(f(0), g0[1]));
        assert(areEqual(f(1), g0[2]));
        assert(areEqual(f(2 - 5 * Eps), g0[3], 10));

        const double x = 0.5;
        const double x2 = x * x;
        const double x3 = x2 * x;
        assert(areEqual(f(x), x3));
        double first;
        assert(areEqual(f(x, &first), x3));
        assert(areEqual(first, 3 * x2));
        double second;
        assert(areEqual(f(x, &first, &second), x3));
        assert(areEqual(first, 3 * x2));
        assert(areEqual(second, 6 * x));
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
        g0[i] = (i - 1.) * (i - 1.);
        g1[i] = 2. * (i - 1.);
        g2[i] = 2.;
      }
      f.setGridValues(g0, g1, g2);
      {
        assert(areEqual(f(-1), g0[0]));
        assert(areEqual(f(0), g0[1]));
        assert(areEqual(f(1), g0[2]));
        assert(areEqual(f(2 - 5 * Eps), g0[3], 10));

        const double x = 0.5;
        const double x2 = x * x;
        assert(areEqual(f(x), x2));
        double first;
        assert(areEqual(f(x, &first), x2));
        assert(areEqual(first, 2 * x));
        double second;
        assert(areEqual(f(x, &first, &second), x2));
        assert(areEqual(first, 2 * x));
        assert(areEqual(second, 2));
      }
    }
  }

  //
  // Example code.
  //

  // Linear.
  {
    typedef numerical::PolynomialInterpolationUsingCoefficients<double, 1> F;
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
    typedef numerical::PolynomialInterpolationUsingCoefficients<double, 3> F;
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
    double derivative;
    for (std::size_t i = 0; i != numberOfCells; ++i) {
      const double x = Dx * (i + 0.5);
      assert(std::abs(f(x) - std::exp(x)) < Dx * Dx);
      // Check the derivative as well.
      assert(std::abs(f(x, &derivative) - std::exp(x)) < std::exp(1) * Dx * Dx);
      assert(std::abs(derivative - std::exp(x)) < std::exp(1) * Dx);
    }
  }
  // Quintic.
  {
    typedef numerical::PolynomialInterpolationUsingCoefficients<double, 5> F;
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
    double first, second;
    for (std::size_t i = 0; i != numberOfCells; ++i) {
      const double x = Dx * (i + 0.5);
      // Check the function value.
      assert(std::abs(f(x) - std::exp(x)) < Dx * Dx);
      // Check the first derivative as well.
      assert(std::abs(f(x, &first) - std::exp(x)) < Dx * Dx);
      assert(std::abs(first - std::exp(x)) < Dx);
      // Check the first and second derivatives.
      assert(std::abs(f(x, &first, &second) - std::exp(x)) < std::exp(1) * Dx * Dx);
      assert(std::abs(first - std::exp(x)) < std::exp(1) * Dx);
      assert(std::abs(second - std::exp(x)) < std::exp(1) * Dx);
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
      typedef numerical::PolynomialInterpolationUsingCoefficients<double, 5>
      F;
      F f(grid.size(), Lower, Upper);
      try {
        f.setGridValues(grid.begin());
      }
      catch (std::exception& e) {
        std::cout << e.what() << '\n';
      }
      for (double x = 0; x < 0.25; x += 0.01) {
        std::cout << x << ' ' << f(x) << '\n';
      }
    }
    {
      std::cout << "\nCubic.\n";
      typedef numerical::PolynomialInterpolationUsingCoefficients<double, 3>
      F;
      F f(grid.size(), Lower, Upper);
      try {
        f.setGridValues(grid.begin());
      }
      catch (std::exception& e) {
        std::cout << e.what() << '\n';
      }
      for (double x = 0; x < 0.25; x += 0.01) {
        std::cout << x << ' ' << f(x) << '\n';
      }
    }
    {
      std::cout << "\nLinear.\n";
      typedef numerical::PolynomialInterpolationUsingCoefficients<double, 1>
      F;
      F f(grid.size(), Lower, Upper);
      try {
        f.setGridValues(grid.begin());
      }
      catch (std::exception& e) {
        std::cout << e.what() << '\n';
      }
      for (double x = 0; x < 0.25; x += 0.01) {
        std::cout << x << ' ' << f(x) << '\n';
      }
    }
  }


  return 0;
}
