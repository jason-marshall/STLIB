// -*- C++ -*-

#include "stlib/numerical/interpolation/MultiArrayOfRegularGrids.h"

#include <cassert>

using namespace stlib;

int
main()
{

  {
    const std::size_t GridDimension = 1;
    const std::size_t ArrayDimension = 1;
    typedef numerical::MultiArrayOfRegularGrids<double, GridDimension,
            ArrayDimension> GridArray;
    typedef GridArray::ArraySizeList ArraySizeList;
    typedef GridArray::GridSizeList GridSizeList;
    typedef GridArray::GridRef GridRef;
    typedef GridArray::GridConstRef GridConstRef;
    typedef container::MultiIndexRange<ArrayDimension> Range;
    typedef container::MultiIndexRangeIterator<ArrayDimension> Iterator;

    const GridSizeList gridExtents = {{10}};
    const ArraySizeList arrayExtents = {{20}};
    const std::size_t gridSize = stlib::ext::product(gridExtents);
    const Range range(arrayExtents);
    const Iterator begin = Iterator::begin(range);
    const Iterator end = Iterator::end(range);
    GridArray x(gridExtents, arrayExtents);
    {
      GridArray y = x;
    }
    // Grid manipulator.
    {
      for (Iterator i = begin; i != end; ++i) {
        GridRef g = x.grid(*i);
        const std::size_t offset = stlib::ext::sum(*i);
        for (std::size_t n = 0; n != g.size(); ++n) {
          g[n] = offset + n;
        }
      }
    }
    // Grid accessor.
    {
      const GridArray& y = x;
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
        double* g = x.gridData(*i);
        const std::size_t offset = stlib::ext::sum(*i);
        for (std::size_t n = 0; n != gridSize; ++n) {
          g[n] = offset * n;
        }
      }
    }
    // Grid accessor.
    {
      const GridArray& y = x;
      for (Iterator i = begin; i != end; ++i) {
        const double* g = y.gridData(*i);
        const std::size_t offset = stlib::ext::sum(*i);
        for (std::size_t n = 0; n != gridSize; ++n) {
          assert(g[n] == offset * n);
        }
      }
    }
  }

  {
    const std::size_t GridDimension = 1;
    const std::size_t ArrayDimension = 2;
    typedef numerical::MultiArrayOfRegularGrids<double, GridDimension,
            ArrayDimension> GridArray;
    typedef GridArray::ArraySizeList ArraySizeList;
    typedef GridArray::GridSizeList GridSizeList;
    typedef GridArray::GridRef GridRef;
    typedef GridArray::GridConstRef GridConstRef;
    typedef container::MultiIndexRange<ArrayDimension> Range;
    typedef container::MultiIndexRangeIterator<ArrayDimension> Iterator;

    const GridSizeList gridExtents = {{10}};
    const ArraySizeList arrayExtents = {{20, 30}};
    const std::size_t gridSize = stlib::ext::product(gridExtents);
    const Range range(arrayExtents);
    const Iterator begin = Iterator::begin(range);
    const Iterator end = Iterator::end(range);
    GridArray x(gridExtents, arrayExtents);
    {
      GridArray y = x;
    }
    // Grid manipulator.
    {
      for (Iterator i = begin; i != end; ++i) {
        GridRef g = x.grid(*i);
        const std::size_t offset = stlib::ext::sum(*i);
        for (std::size_t n = 0; n != g.size(); ++n) {
          g[n] = offset + n;
        }
      }
    }
    // Grid accessor.
    {
      const GridArray& y = x;
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
        double* g = x.gridData(*i);
        const std::size_t offset = stlib::ext::sum(*i);
        for (std::size_t n = 0; n != gridSize; ++n) {
          g[n] = offset * n;
        }
      }
    }
    // Grid accessor.
    {
      const GridArray& y = x;
      for (Iterator i = begin; i != end; ++i) {
        const double* g = y.gridData(*i);
        const std::size_t offset = stlib::ext::sum(*i);
        for (std::size_t n = 0; n != gridSize; ++n) {
          assert(g[n] == offset * n);
        }
      }
    }
  }

  {
    const std::size_t GridDimension = 2;
    const std::size_t ArrayDimension = 1;
    typedef numerical::MultiArrayOfRegularGrids<double, GridDimension,
            ArrayDimension> GridArray;
    typedef GridArray::ArraySizeList ArraySizeList;
    typedef GridArray::GridSizeList GridSizeList;
    typedef GridArray::GridRef GridRef;
    typedef GridArray::GridConstRef GridConstRef;
    typedef container::MultiIndexRange<ArrayDimension> Range;
    typedef container::MultiIndexRangeIterator<ArrayDimension> Iterator;

    const GridSizeList gridExtents = {{10, 20}};
    const ArraySizeList arrayExtents = {{20}};
    const std::size_t gridSize = stlib::ext::product(gridExtents);
    const Range range(arrayExtents);
    const Iterator begin = Iterator::begin(range);
    const Iterator end = Iterator::end(range);
    GridArray x(gridExtents, arrayExtents);
    {
      GridArray y = x;
    }
    // Grid manipulator.
    {
      for (Iterator i = begin; i != end; ++i) {
        GridRef g = x.grid(*i);
        const std::size_t offset = stlib::ext::sum(*i);
        for (std::size_t n = 0; n != g.size(); ++n) {
          g[n] = offset + n;
        }
      }
    }
    // Grid accessor.
    {
      const GridArray& y = x;
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
        double* g = x.gridData(*i);
        const std::size_t offset = stlib::ext::sum(*i);
        for (std::size_t n = 0; n != gridSize; ++n) {
          g[n] = offset * n;
        }
      }
    }
    // Grid accessor.
    {
      const GridArray& y = x;
      for (Iterator i = begin; i != end; ++i) {
        const double* g = y.gridData(*i);
        const std::size_t offset = stlib::ext::sum(*i);
        for (std::size_t n = 0; n != gridSize; ++n) {
          assert(g[n] == offset * n);
        }
      }
    }
  }

  {
    const std::size_t GridDimension = 2;
    const std::size_t ArrayDimension = 2;
    typedef numerical::MultiArrayOfRegularGrids<double, GridDimension,
            ArrayDimension> GridArray;
    typedef GridArray::ArraySizeList ArraySizeList;
    typedef GridArray::GridSizeList GridSizeList;
    typedef GridArray::GridRef GridRef;
    typedef GridArray::GridConstRef GridConstRef;
    typedef container::MultiIndexRange<ArrayDimension> Range;
    typedef container::MultiIndexRangeIterator<ArrayDimension> Iterator;

    const GridSizeList gridExtents = {{2, 3}};
    const ArraySizeList arrayExtents = {{5, 7}};
    const std::size_t gridSize = stlib::ext::product(gridExtents);
    const Range range(arrayExtents);
    const Iterator begin = Iterator::begin(range);
    const Iterator end = Iterator::end(range);
    GridArray x(gridExtents, arrayExtents);
    {
      GridArray y = x;
    }
    // Grid manipulator.
    {
      for (Iterator i = begin; i != end; ++i) {
        GridRef g = x.grid(*i);
        const std::size_t offset = stlib::ext::sum(*i);
        for (std::size_t n = 0; n != g.size(); ++n) {
          g[n] = offset + n;
        }
      }
    }
    // Grid accessor.
    {
      const GridArray& y = x;
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
        double* g = x.gridData(*i);
        const std::size_t offset = stlib::ext::sum(*i);
        for (std::size_t n = 0; n != gridSize; ++n) {
          g[n] = offset * n;
        }
      }
    }
    // Grid accessor.
    {
      const GridArray& y = x;
      for (Iterator i = begin; i != end; ++i) {
        const double* g = y.gridData(*i);
        const std::size_t offset = stlib::ext::sum(*i);
        for (std::size_t n = 0; n != gridSize; ++n) {
          assert(g[n] == offset * n);
        }
      }
    }
  }

  return 0;
}
