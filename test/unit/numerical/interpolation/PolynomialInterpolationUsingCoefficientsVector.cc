// -*- C++ -*-

#include "stlib/numerical/interpolation/PolynomialInterpolationUsingCoefficientsVector.h"
#include "stlib/numerical/equality.h"
#include "stlib/container/MultiIndexRangeIterator.h"

using namespace stlib;

int
main()
{

  {
    const std::size_t Order = 1;
    typedef numerical::PolynomialInterpolationUsingCoefficientsVector
    <double, Order> F;
    typedef F::Value Value;

    const std::size_t GridSize = 10;
    const double Lower = 2;
    const double Upper = 3;
    const std::size_t NumberOfGrids = 20;
    F f(GridSize, Lower, Upper, NumberOfGrids);

    {
      F y = f;
    }
    // Grid data manipulator.
    {
      std::vector<Value> data(GridSize);
      for (std::size_t i = 0; i != f.getNumberOfGrids(); ++i) {
        const std::size_t offset = i;
        for (std::size_t n = 0; n != GridSize; ++n) {
          data[n] = offset + n;
        }
        f.setGrid(i);
        f.setGridValues(data.begin());
      }
    }
    // Interpolation.
    {
      for (std::size_t i = 0; i != f.getNumberOfGrids(); ++i) {
        f.setGrid(i);
        const std::size_t offset = i;
        assert(numerical::areEqual(f(Lower), offset));
      }
    }
  }

  {
    const std::size_t Order = 3;
    typedef numerical::PolynomialInterpolationUsingCoefficientsVector
    <double, Order> F;
    typedef F::Value Value;

    const std::size_t GridSize = 10;
    const double Lower = 2;
    const double Upper = 3;
    const std::size_t NumberOfGrids = 20;
    F f(GridSize, Lower, Upper, NumberOfGrids);

    {
      F y = f;
    }
    // Grid data manipulator.
    {
      std::vector<Value> data(GridSize);
      for (std::size_t i = 0; i != f.getNumberOfGrids(); ++i) {
        const std::size_t offset = i;
        for (std::size_t n = 0; n != GridSize; ++n) {
          data[n] = offset + n;
        }
        f.setGrid(i);
        f.setGridValues(data.begin());
      }
    }
    // Interpolation.
    {
      for (std::size_t i = 0; i != f.getNumberOfGrids(); ++i) {
        f.setGrid(i);
        const std::size_t offset = i;
        assert(numerical::areEqual(f(Lower), offset));
      }
    }
  }

  //
  // Example code.
  //

  // Linear.
  {
    // Linear interpolation with a vector of grids.
    typedef numerical::PolynomialInterpolationUsingCoefficientsVector<double, 1> F;
    // Make a grid to sample the exponential function on the domain [0..1).
    // Use a grid spacing of 0.1.
    std::vector<double> grid(11);
    const double Lower = 0;
    const double Upper = 1;
    const double Dx = (Upper - Lower) / (grid.size() - 1);
    const std::size_t numberOfGrids = 20;
    // Construct the interpolating function.
    F f(grid.size(), Lower, Upper, numberOfGrids);
    // Set values for each of the grids.
    // Loop over the grids in the vector.
    for (std::size_t n = 0; n != f.getNumberOfGrids(); ++n) {
      // Select a grid.
      f.setGrid(n);
      // A different offset for each grid.
      const double offset = n;
      // Set the grid values.
      for (std::size_t i = 0; i != grid.size(); ++i) {
        grid[i] = offset + std::exp(Dx * i);
      }
      f.setGridValues(grid.begin());
    }
    // Check that the function has the correct values at the grid points.
    const std::size_t numberOfCells = grid.size() - 1;
    for (std::size_t n = 0; n != f.getNumberOfGrids(); ++n) {
      f.setGrid(n);
      const double offset = n;
      for (std::size_t i = 0; i != numberOfCells; ++i) {
        assert(numerical::areEqual(f(Dx * i), offset + std::exp(Dx * i)));
      }
    }
  }

  return 0;
}
