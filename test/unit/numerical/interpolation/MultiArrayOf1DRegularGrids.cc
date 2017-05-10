// -*- C++ -*-

#include "stlib/numerical/interpolation/MultiArrayOf1DRegularGrids.h"
#include "stlib/container/MultiIndexRangeIterator.h"

#include <cassert>

using namespace stlib;

template<typename _T, std::size_t _Order, std::size_t _Dimension>
class MultiArrayOf1DRegularGridsTest :
  public numerical::MultiArrayOf1DRegularGrids
  <_T, _Order, _Dimension>
{
  //
  // Types.
  //
private:

  typedef numerical::MultiArrayOf1DRegularGrids
  <_T, _Order, _Dimension> Base;

public:

  //! The (multi) size type for the array.
  typedef typename Base::SizeList SizeList;

  //--------------------------------------------------------------------------
  /*! \name Constructors etc.
    We use the default copy constructor and destructor. The assignment
    operator is not implemented.
  */
  //! @{
public:

  //! Construct from the grid size and the multi-array extents.
  MultiArrayOf1DRegularGridsTest
  (const std::size_t& gridSize, const SizeList& arrayExtents) :
    Base(gridSize, arrayExtents)
  {
  }

private:

  // The assignment operator is not implemented.
  MultiArrayOf1DRegularGridsTest&
  operator=(const MultiArrayOf1DRegularGridsTest&);

  //! @}
  //--------------------------------------------------------------------------
  //! \name Manipulators.
  //! @{
public:

  using Base::_gridData;

  //! @}
};

int
main()
{

  {
    const std::size_t Order = 1;
    const std::size_t Dimension = 1;
    typedef MultiArrayOf1DRegularGridsTest<double, Order, Dimension>
    GridArray;
    typedef GridArray::SizeList SizeList;
    typedef GridArray::Value Value;
    typedef container::MultiIndexRange<Dimension> Range;
    typedef container::MultiIndexRangeIterator<Dimension> Iterator;

    const std::size_t gridSize = 10;
    const SizeList arrayExtents = {{20}};
    GridArray x(gridSize, arrayExtents);

    {
      GridArray y = x;
    }
    // Grid data manipulator.
    const Range range(arrayExtents);
    const Iterator begin = Iterator::begin(range);
    const Iterator end = Iterator::end(range);
    {
      for (Iterator i = begin; i != end; ++i) {
        Value* g = x._gridData(*i);
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
        const Value* g = y.gridData(*i);
        const std::size_t offset = stlib::ext::sum(*i);
        for (std::size_t n = 0; n != gridSize; ++n) {
          assert(g[n] == offset * n);
        }
      }
    }
  }

  {
    const std::size_t Order = 1;
    const std::size_t Dimension = 2;
    typedef MultiArrayOf1DRegularGridsTest<double, Order, Dimension>
    GridArray;
    typedef GridArray::SizeList SizeList;
    typedef GridArray::Value Value;
    typedef container::MultiIndexRange<Dimension> Range;
    typedef container::MultiIndexRangeIterator<Dimension> Iterator;

    const std::size_t gridSize = 10;
    const SizeList arrayExtents = {{20, 30}};
    GridArray x(gridSize, arrayExtents);

    {
      GridArray y = x;
    }
    // Grid data manipulator.
    const Range range(arrayExtents);
    const Iterator begin = Iterator::begin(range);
    const Iterator end = Iterator::end(range);
    {
      for (Iterator i = begin; i != end; ++i) {
        Value* g = x._gridData(*i);
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
        const Value* g = y.gridData(*i);
        const std::size_t offset = stlib::ext::sum(*i);
        for (std::size_t n = 0; n != gridSize; ++n) {
          assert(g[n] == offset * n);
        }
      }
    }
  }

  {
    const std::size_t Order = 3;
    const std::size_t Dimension = 1;
    typedef MultiArrayOf1DRegularGridsTest<double, Order, Dimension>
    GridArray;
    typedef GridArray::SizeList SizeList;
    typedef GridArray::Value Value;
    typedef container::MultiIndexRange<Dimension> Range;
    typedef container::MultiIndexRangeIterator<Dimension> Iterator;

    const std::size_t gridSize = 10;
    const SizeList arrayExtents = {{20}};
    GridArray x(gridSize, arrayExtents);

    {
      GridArray y = x;
    }
    // Grid data manipulator.
    const Range range(arrayExtents);
    const Iterator begin = Iterator::begin(range);
    const Iterator end = Iterator::end(range);
    {
      for (Iterator i = begin; i != end; ++i) {
        Value* g = x._gridData(*i);
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
        const Value* g = y.gridData(*i);
        const std::size_t offset = stlib::ext::sum(*i);
        for (std::size_t n = 0; n != gridSize; ++n) {
          assert(g[n] == offset * n);
        }
      }
    }
  }

  {
    const std::size_t Order = 3;
    const std::size_t Dimension = 2;
    typedef MultiArrayOf1DRegularGridsTest<double, Order, Dimension>
    GridArray;
    typedef GridArray::SizeList SizeList;
    typedef GridArray::Value Value;
    typedef container::MultiIndexRange<Dimension> Range;
    typedef container::MultiIndexRangeIterator<Dimension> Iterator;

    const std::size_t gridSize = 10;
    const SizeList arrayExtents = {{20, 30}};
    GridArray x(gridSize, arrayExtents);

    {
      GridArray y = x;
    }
    // Grid data manipulator.
    const Range range(arrayExtents);
    const Iterator begin = Iterator::begin(range);
    const Iterator end = Iterator::end(range);
    {
      for (Iterator i = begin; i != end; ++i) {
        Value* g = x._gridData(*i);
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
        const Value* g = y.gridData(*i);
        const std::size_t offset = stlib::ext::sum(*i);
        for (std::size_t n = 0; n != gridSize; ++n) {
          assert(g[n] == offset * n);
        }
      }
    }
  }

  return 0;
}
