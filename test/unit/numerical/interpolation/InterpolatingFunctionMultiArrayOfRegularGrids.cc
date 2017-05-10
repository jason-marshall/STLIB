// -*- C++ -*-

#include "stlib/numerical/interpolation/InterpolatingFunctionMultiArrayOfRegularGrids.h"
#include "stlib/numerical/equality.h"

#include <cassert>

USING_STLIB_EXT_ARRAY_MATH_OPERATORS;
using namespace stlib;

int
main()
{

  {
    const std::size_t GridDimension = 1;
    const std::size_t ArrayDimension = 1;
    const std::size_t DefaultOrder = 1;
    typedef numerical::InterpolatingFunctionMultiArrayOfRegularGrids
    <double, GridDimension, ArrayDimension, DefaultOrder> Functor;
    typedef Functor::ArraySizeList ArraySizeList;
    typedef Functor::GridSizeList GridSizeList;
    typedef Functor::GridRef GridRef;
    typedef Functor::GridConstRef GridConstRef;
    typedef Functor::BBox BBox;
    typedef container::MultiIndexRange<ArrayDimension> Range;
    typedef container::MultiIndexRangeIterator<ArrayDimension> Iterator;
    const GridSizeList gridExtents = {{10}};
    const std::size_t gridSize = stlib::ext::product(gridExtents);
    const BBox domain = {{{0.}}, {{1.}}};
    const ArraySizeList arrayExtents = {{20}};
    const Range range(arrayExtents);
    const Iterator begin = Iterator::begin(range);
    const Iterator end = Iterator::end(range);
    Functor f(gridExtents, domain, arrayExtents);
    {
      Functor g = f;
    }
    // Grid manipulator.
    {
      for (Iterator i = begin; i != end; ++i) {
        GridRef g = f.grid(*i);
        const std::size_t offset = stlib::ext::sum(*i);
        for (std::size_t n = 0; n != g.size(); ++n) {
          g[n] = offset + n;
        }
      }
    }
    // Grid accessor.
    {
      const Functor& y = f;
      for (Iterator i = begin; i != end; ++i) {
        const GridConstRef& g = y.grid(*i);
        const std::size_t offset = stlib::ext::sum(*i);
        for (std::size_t n = 0; n != g.size(); ++n) {
          assert(g[n] == offset + n);
        }
      }
    }
    // Grid data manipulator.
    {
      for (Iterator i = begin; i != end; ++i) {
        double* g = f.gridData(*i);
        const std::size_t offset = stlib::ext::sum(*i);
        for (std::size_t n = 0; n != gridSize; ++n) {
          g[n] = offset * n;
        }
      }
    }
    // Grid accessor.
    {
      const Functor& y = f;
      for (Iterator i = begin; i != end; ++i) {
        const double* g = y.gridData(*i);
        const std::size_t offset = stlib::ext::sum(*i);
        for (std::size_t n = 0; n != gridSize; ++n) {
          assert(g[n] == offset * n);
        }
      }
    }
    // Interpolation.
    {
      for (Iterator i = begin; i != end; ++i) {
        f.setGrid(*i);
        const double* g = f.gridData(*i);
        assert(numerical::areEqual(f(domain.lower), g[0]));
        assert(numerical::areEqual(f(domain.upper),
                                   g[gridSize - 1]));
      }
    }
  }

  {
    const std::size_t GridDimension = 1;
    const std::size_t ArrayDimension = 2;
    const std::size_t DefaultOrder = 1;
    typedef numerical::InterpolatingFunctionMultiArrayOfRegularGrids
    <double, GridDimension, ArrayDimension, DefaultOrder> Functor;
    typedef Functor::ArraySizeList ArraySizeList;
    typedef Functor::GridSizeList GridSizeList;
    typedef Functor::GridRef GridRef;
    typedef Functor::GridConstRef GridConstRef;
    typedef Functor::BBox BBox;
    typedef container::MultiIndexRange<ArrayDimension> Range;
    typedef container::MultiIndexRangeIterator<ArrayDimension> Iterator;

    const GridSizeList gridExtents = {{2}};
    const BBox domain = {{{0.}}, {{1.}}};
    const ArraySizeList arrayExtents = {{3, 5}};
    const std::size_t gridSize = stlib::ext::product(gridExtents);
    const Range range(arrayExtents);
    const Iterator begin = Iterator::begin(range);
    const Iterator end = Iterator::end(range);
    Functor f(gridExtents, domain, arrayExtents);
    {
      Functor g = f;
    }
    // Grid manipulator.
    {
      for (Iterator i = begin; i != end; ++i) {
        GridRef g = f.grid(*i);
        const std::size_t offset = stlib::ext::sum(*i);
        for (std::size_t n = 0; n != g.size(); ++n) {
          g[n] = offset + n;
        }
      }
    }
    // Grid accessor.
    {
      const Functor& y = f;
      for (Iterator i = begin; i != end; ++i) {
        const GridConstRef& g = y.grid(*i);
        const std::size_t offset = stlib::ext::sum(*i);
        for (std::size_t n = 0; n != g.size(); ++n) {
          assert(g[n] == offset + n);
        }
      }
    }
    // Grid data manipulator.
    {
      for (Iterator i = begin; i != end; ++i) {
        double* g = f.gridData(*i);
        const std::size_t offset = stlib::ext::sum(*i);
        for (std::size_t n = 0; n != gridSize; ++n) {
          g[n] = offset * n;
        }
      }
    }
    // Grid accessor.
    {
      const Functor& y = f;
      for (Iterator i = begin; i != end; ++i) {
        const double* g = y.gridData(*i);
        const std::size_t offset = stlib::ext::sum(*i);
        for (std::size_t n = 0; n != gridSize; ++n) {
          assert(g[n] == offset * n);
        }
      }
    }
    // Interpolation.
    {
      for (Iterator i = begin; i != end; ++i) {
        f.setGrid(*i);
        const double* g = f.gridData(*i);
        assert(numerical::areEqual(f(domain.lower), g[0]));
        assert(numerical::areEqual(f(domain.upper),
                                   g[gridSize - 1]));
      }
    }
  }

  {
    const std::size_t GridDimension = 2;
    const std::size_t ArrayDimension = 1;
    const std::size_t DefaultOrder = 1;
    typedef numerical::InterpolatingFunctionMultiArrayOfRegularGrids
    <double, GridDimension, ArrayDimension, DefaultOrder> Functor;
    typedef Functor::ArraySizeList ArraySizeList;
    typedef Functor::GridSizeList GridSizeList;
    typedef Functor::GridRef GridRef;
    typedef Functor::GridConstRef GridConstRef;
    typedef Functor::BBox BBox;
    typedef container::MultiIndexRange<ArrayDimension> Range;
    typedef container::MultiIndexRangeIterator<ArrayDimension> Iterator;

    const GridSizeList gridExtents = {{2, 3}};
    const BBox domain = {{{0., 0.}}, {{1., 1.}}};
    const ArraySizeList arrayExtents = {{5}};
    const std::size_t gridSize = stlib::ext::product(gridExtents);
    const Range range(arrayExtents);
    const Iterator begin = Iterator::begin(range);
    const Iterator end = Iterator::end(range);
    Functor f(gridExtents, domain, arrayExtents);
    {
      Functor g = f;
    }
    // Grid manipulator.
    {
      for (Iterator i = begin; i != end; ++i) {
        GridRef g = f.grid(*i);
        const std::size_t offset = stlib::ext::sum(*i);
        for (std::size_t n = 0; n != g.size(); ++n) {
          g[n] = offset + n;
        }
      }
    }
    // Grid accessor.
    {
      const Functor& y = f;
      for (Iterator i = begin; i != end; ++i) {
        const GridConstRef& g = y.grid(*i);
        const std::size_t offset = stlib::ext::sum(*i);
        for (std::size_t n = 0; n != g.size(); ++n) {
          assert(g[n] == offset + n);
        }
      }
    }
    // Grid data manipulator.
    {
      for (Iterator i = begin; i != end; ++i) {
        double* g = f.gridData(*i);
        const std::size_t offset = stlib::ext::sum(*i);
        for (std::size_t n = 0; n != gridSize; ++n) {
          g[n] = offset * n;
        }
      }
    }
    // Grid accessor.
    {
      const Functor& y = f;
      for (Iterator i = begin; i != end; ++i) {
        const double* g = y.gridData(*i);
        const std::size_t offset = stlib::ext::sum(*i);
        for (std::size_t n = 0; n != gridSize; ++n) {
          assert(g[n] == offset * n);
        }
      }
    }
    // Interpolation.
    {
      for (Iterator i = begin; i != end; ++i) {
        f.setGrid(*i);
        const double* g = f.gridData(*i);
        assert(numerical::areEqual(f(domain.lower), g[0]));
        assert(numerical::areEqual(f(domain.upper),
                                   g[gridSize - 1]));
      }
    }
  }

  {
    const std::size_t GridDimension = 2;
    const std::size_t ArrayDimension = 2;
    const std::size_t DefaultOrder = 1;
    typedef numerical::InterpolatingFunctionMultiArrayOfRegularGrids
    <double, GridDimension, ArrayDimension, DefaultOrder> Functor;
    typedef Functor::ArraySizeList ArraySizeList;
    typedef Functor::GridSizeList GridSizeList;
    typedef Functor::GridRef GridRef;
    typedef Functor::GridConstRef GridConstRef;
    typedef Functor::BBox BBox;
    typedef container::MultiIndexRange<ArrayDimension> Range;
    typedef container::MultiIndexRangeIterator<ArrayDimension> Iterator;

    const GridSizeList gridExtents = {{2, 3}};
    const BBox domain = {{{0., 0.}}, {{1., 1.}}};
    const ArraySizeList arrayExtents = {{5, 7}};
    const std::size_t gridSize = stlib::ext::product(gridExtents);
    const Range range(arrayExtents);
    const Iterator begin = Iterator::begin(range);
    const Iterator end = Iterator::end(range);
    Functor f(gridExtents, domain, arrayExtents);
    {
      Functor g = f;
    }
    // Grid manipulator.
    {
      for (Iterator i = begin; i != end; ++i) {
        GridRef g = f.grid(*i);
        const std::size_t offset = stlib::ext::sum(*i);
        for (std::size_t n = 0; n != g.size(); ++n) {
          g[n] = offset + n;
        }
      }
    }
    // Grid accessor.
    {
      const Functor& y = f;
      for (Iterator i = begin; i != end; ++i) {
        const GridConstRef& g = y.grid(*i);
        const std::size_t offset = stlib::ext::sum(*i);
        for (std::size_t n = 0; n != g.size(); ++n) {
          assert(g[n] == offset + n);
        }
      }
    }
    // Grid data manipulator.
    {
      for (Iterator i = begin; i != end; ++i) {
        double* g = f.gridData(*i);
        const std::size_t offset = stlib::ext::sum(*i);
        for (std::size_t n = 0; n != gridSize; ++n) {
          g[n] = offset * n;
        }
      }
    }
    // Grid accessor.
    {
      const Functor& y = f;
      for (Iterator i = begin; i != end; ++i) {
        const double* g = y.gridData(*i);
        const std::size_t offset = stlib::ext::sum(*i);
        for (std::size_t n = 0; n != gridSize; ++n) {
          assert(g[n] == offset * n);
        }
      }
    }
    // Interpolation.
    {
      for (Iterator i = begin; i != end; ++i) {
        f.setGrid(*i);
        const double* g = f.gridData(*i);
        assert(numerical::areEqual(f(domain.lower), g[0]));
        assert(numerical::areEqual(f(domain.upper),
                                   g[gridSize - 1]));
      }
    }
  }

  // Example code.
  {
    // A 3-D array of 2-D grids.
    const std::size_t GridDimension = 2;
    const std::size_t ArrayDimension = 3;
    const std::size_t DefaultOrder = 1;
    typedef numerical::InterpolatingFunctionMultiArrayOfRegularGrids
    <double, GridDimension, ArrayDimension, DefaultOrder>
    InterpolatingFunction;
    typedef InterpolatingFunction::ArraySizeList ArraySizeList;
    typedef InterpolatingFunction::GridSizeList GridSizeList;
    typedef InterpolatingFunction::GridRef GridRef;
    typedef InterpolatingFunction::Point Point;
    typedef InterpolatingFunction::BBox BBox;
    typedef container::MultiIndexRange<ArrayDimension> ArrayRange;
    typedef container::MultiIndexRangeIterator<ArrayDimension> ArrayIterator;
    typedef container::MultiIndexRange<GridDimension> GridRange;
    typedef container::MultiIndexRangeIterator<GridDimension> GridIterator;

    const GridSizeList gridExtents = {{2, 3}};
    // The Cartesian domain is (0..2)x(0..3).
    const Point lower = {{0, 0}};
    const Point upper = {{2, 3}};
    const Point dx = (upper - lower) /
      ext::convert_array<double>(gridExtents - std::size_t(1));
    const BBox domain = {lower, upper};
    const ArraySizeList arrayExtents = {{5, 7, 11}};
    InterpolatingFunction f(gridExtents, domain, arrayExtents);

    // Useful for iterating over the grids.
    const ArrayRange arrayRange(arrayExtents);
    const ArrayIterator arrayBegin = ArrayIterator::begin(arrayRange);
    const ArrayIterator arrayEnd = ArrayIterator::end(arrayRange);
    const GridRange gridRange(gridExtents);
    const GridIterator gridBegin = GridIterator::begin(gridRange);
    const GridIterator gridEnd = GridIterator::end(gridRange);

    // Set the array values to the sum of the point coordinates plus the
    // sum of the grid indices.
    Point x;
    // Loop over the grids in the array.
    for (ArrayIterator a = arrayBegin; a != arrayEnd; ++a) {
      GridRef grid = f.grid(*a);
      const double offset = stlib::ext::sum(*a);
      // Loop over the elements in the grid.
      for (GridIterator g = gridBegin; g != gridEnd; ++g) {
        x = lower + stlib::ext::convert_array<double>(*g) * dx;
        grid(*g) = stlib::ext::sum(x) + offset;
      }
    }

    // Check that the function has the correct values at the grid points.
    // Loop over the grids in the array.
    for (ArrayIterator a = arrayBegin; a != arrayEnd; ++a) {
      // Select the grid.
      f.setGrid(*a);
      const double offset = stlib::ext::sum(*a);
      // Loop over the elements in the grid.
      for (GridIterator g = gridBegin; g != gridEnd; ++g) {
        x = lower + stlib::ext::convert_array<double>(*g) * dx;
        assert(numerical::areEqual(f(x), stlib::ext::sum(x) + offset));
      }
    }
  }

  return 0;
}
