// -*- C++ -*-

#include "stlib/numerical/interpolation/InterpolatingFunctionMultiArrayOf1DRegularGrids.h"
#include "stlib/numerical/equality.h"
#include "stlib/container/MultiIndexRangeIterator.h"

using namespace stlib;

int
main()
{

  {
    const std::size_t Order = 1;
    const std::size_t Dimension = 1;
    typedef numerical::InterpolatingFunctionMultiArrayOf1DRegularGrids
    <double, Order, Dimension> F;
    typedef F::SizeList SizeList;
    typedef F::Value Value;
    typedef container::MultiIndexRange<Dimension> Range;
    typedef container::MultiIndexRangeIterator<Dimension> Iterator;

    const std::size_t GridSize = 10;
    const double Lower = 2;
    const double Upper = 3;
    const SizeList ArrayExtents = {{20}};
    F f(GridSize, Lower, Upper, ArrayExtents);

    {
      F y = f;
    }
    // Grid data manipulator.
    const Range range(ArrayExtents);
    const Iterator begin = Iterator::begin(range);
    const Iterator end = Iterator::end(range);
    {
      std::vector<Value> data(GridSize);
      for (Iterator i = begin; i != end; ++i) {
        const std::size_t offset = stlib::ext::sum(*i);
        for (std::size_t n = 0; n != GridSize; ++n) {
          data[n] = offset + n;
        }
        f.setGrid(*i);
        f.setGridValues(data.begin(), data.end());
      }
    }
    // Grid accessor.
    {
      const F& y = f;
      for (Iterator i = begin; i != end; ++i) {
        const Value* g = y.gridData(*i);
        const std::size_t offset = stlib::ext::sum(*i);
        for (std::size_t n = 0; n != GridSize; ++n) {
          assert(g[n] == offset + n);
        }
      }
    }
    // Interpolation.
    {
      for (Iterator i = begin; i != end; ++i) {
        f.setGrid(*i);
        const Value* g = f.gridData(*i);
        assert(numerical::areEqual(f(Lower), g[0]));
      }
    }
  }

  {
    const std::size_t Order = 1;
    const std::size_t Dimension = 2;
    typedef numerical::InterpolatingFunctionMultiArrayOf1DRegularGrids
    <double, Order, Dimension> F;
    typedef F::SizeList SizeList;
    typedef F::Value Value;
    typedef container::MultiIndexRange<Dimension> Range;
    typedef container::MultiIndexRangeIterator<Dimension> Iterator;

    const std::size_t GridSize = 10;
    const double Lower = 2;
    const double Upper = 3;
    const SizeList ArrayExtents = {{20, 30}};
    F f(GridSize, Lower, Upper, ArrayExtents);

    {
      F y = f;
    }
    // Grid data manipulator.
    const Range range(ArrayExtents);
    const Iterator begin = Iterator::begin(range);
    const Iterator end = Iterator::end(range);
    {
      std::vector<Value> data(GridSize);
      for (Iterator i = begin; i != end; ++i) {
        const std::size_t offset = stlib::ext::sum(*i);
        for (std::size_t n = 0; n != GridSize; ++n) {
          data[n] = offset + n;
        }
        f.setGrid(*i);
        f.setGridValues(data.begin(), data.end());
      }
    }
    // Grid accessor.
    {
      const F& y = f;
      for (Iterator i = begin; i != end; ++i) {
        const Value* g = y.gridData(*i);
        const std::size_t offset = stlib::ext::sum(*i);
        for (std::size_t n = 0; n != GridSize; ++n) {
          assert(g[n] == offset + n);
        }
      }
    }
    // Interpolation.
    {
      for (Iterator i = begin; i != end; ++i) {
        f.setGrid(*i);
        const Value* g = f.gridData(*i);
        assert(numerical::areEqual(f(Lower), g[0]));
      }
    }
  }

  {
    const std::size_t Order = 3;
    const std::size_t Dimension = 1;
    typedef numerical::InterpolatingFunctionMultiArrayOf1DRegularGrids
    <double, Order, Dimension> F;
    typedef F::SizeList SizeList;
    typedef F::Value Value;
    typedef container::MultiIndexRange<Dimension> Range;
    typedef container::MultiIndexRangeIterator<Dimension> Iterator;

    const std::size_t GridSize = 10;
    const double Lower = 2;
    const double Upper = 3;
    const SizeList ArrayExtents = {{20}};
    F f(GridSize, Lower, Upper, ArrayExtents);

    {
      F y = f;
    }
    // Grid data manipulator.
    const Range range(ArrayExtents);
    const Iterator begin = Iterator::begin(range);
    const Iterator end = Iterator::end(range);
    {
      std::vector<Value> data(GridSize);
      for (Iterator i = begin; i != end; ++i) {
        const std::size_t offset = stlib::ext::sum(*i);
        for (std::size_t n = 0; n != GridSize; ++n) {
          data[n] = offset + n;
        }
        f.setGrid(*i);
        f.setGridValues(data.begin(), data.end());
      }
    }
    // Grid accessor.
    {
      const F& y = f;
      for (Iterator i = begin; i != end; ++i) {
        const Value* g = y.gridData(*i);
        const std::size_t offset = stlib::ext::sum(*i);
        for (std::size_t n = 0; n != GridSize; ++n) {
          assert(g[n] == offset + n);
        }
      }
    }
    // Interpolation.
    {
      for (Iterator i = begin; i != end; ++i) {
        f.setGrid(*i);
        const Value* g = f.gridData(*i);
        assert(numerical::areEqual(f(Lower), g[0]));
      }
    }
  }

  {
    const std::size_t Order = 3;
    const std::size_t Dimension = 2;
    typedef numerical::InterpolatingFunctionMultiArrayOf1DRegularGrids
    <double, Order, Dimension> F;
    typedef F::SizeList SizeList;
    typedef F::Value Value;
    typedef container::MultiIndexRange<Dimension> Range;
    typedef container::MultiIndexRangeIterator<Dimension> Iterator;

    const std::size_t GridSize = 10;
    const double Lower = 2;
    const double Upper = 3;
    const SizeList ArrayExtents = {{20, 30}};
    F f(GridSize, Lower, Upper, ArrayExtents);

    {
      F y = f;
    }
    // Grid data manipulator.
    const Range range(ArrayExtents);
    const Iterator begin = Iterator::begin(range);
    const Iterator end = Iterator::end(range);
    {
      std::vector<Value> data(GridSize);
      for (Iterator i = begin; i != end; ++i) {
        const std::size_t offset = stlib::ext::sum(*i);
        for (std::size_t n = 0; n != GridSize; ++n) {
          data[n] = offset + n;
        }
        f.setGrid(*i);
        f.setGridValues(data.begin(), data.end());
      }
    }
    // Grid accessor.
    {
      const F& y = f;
      for (Iterator i = begin; i != end; ++i) {
        const Value* g = y.gridData(*i);
        const std::size_t offset = stlib::ext::sum(*i);
        for (std::size_t n = 0; n != GridSize; ++n) {
          assert(g[n] == offset + n);
        }
      }
    }
    // Interpolation.
    {
      for (Iterator i = begin; i != end; ++i) {
        f.setGrid(*i);
        const Value* g = f.gridData(*i);
        assert(numerical::areEqual(f(Lower), g[0]));
      }
    }
  }

  //
  // Example code.
  //

  // Linear.
  {
    // Linear interpolation. 2-D array of grids.
    const std::size_t Dimension = 2;
    typedef numerical::InterpolatingFunctionMultiArrayOf1DRegularGrids<double, 1, Dimension>
    F;
    typedef F::SizeList SizeList;
    typedef container::MultiIndexRange<Dimension> ArrayRange;
    typedef container::MultiIndexRangeIterator<Dimension> ArrayIterator;
    // Make a grid to sample the exponential function on the domain [0..1).
    // Use a grid spacing of 0.1.
    std::size_t gridSize = 11;
    const double Lower = 0;
    const double Upper = 1;
    const double Dx = (Upper - Lower) / (gridSize - 1);
    const SizeList arrayExtents = {{10, 20}};
    // Construct the interpolating function.
    F f(gridSize, Lower, Upper, arrayExtents);
    // Set values of each of the grids.
    const ArrayRange arrayRange(arrayExtents);
    const ArrayIterator arrayBegin = ArrayIterator::begin(arrayRange);
    const ArrayIterator arrayEnd = ArrayIterator::end(arrayRange);
    std::vector<double> grid(gridSize);
    // Loop over the grids in the array.
    for (ArrayIterator a = arrayBegin; a != arrayEnd; ++a) {
      // Select a grid.
      f.setGrid(*a);
      // A different offset for each grid.
      const double offset = stlib::ext::sum(*a);
      // Set the grid values.
      for (std::size_t i = 0; i != grid.size(); ++i) {
        grid[i] = offset + std::exp(Dx * i);
      }
      f.setGridValues(grid.begin(), grid.end());
    }
    // Check that the function has the correct values at the grid points.
    const std::size_t numberOfCells = grid.size() - 1;
    for (ArrayIterator a = arrayBegin; a != arrayEnd; ++a) {
      f.setGrid(*a);
      const double offset = stlib::ext::sum(*a);
      for (std::size_t i = 0; i != numberOfCells; ++i) {
        assert(numerical::areEqual(f(Dx * i), offset + std::exp(Dx * i)));
      }
    }
  }

  return 0;
}
