// -*- C++ -*-

#include "stlib/numerical/interpolation/MultiArrayOfCoefficientArrays.h"
#include "stlib/container/MultiIndexRangeIterator.h"

#include <cassert>

using namespace stlib;

template<typename _T, std::size_t _Order, std::size_t _Dimension>
class MultiArrayOfCoefficientArraysTest :
  public numerical::MultiArrayOfCoefficientArrays<_T, _Order, _Dimension>
{
  //
  // Types.
  //
private:

  typedef numerical::MultiArrayOfCoefficientArrays<_T, _Order, _Dimension>
  Base;

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
  MultiArrayOfCoefficientArraysTest
  (const std::size_t& gridSize, const SizeList& arrayExtents) :
    Base(gridSize, arrayExtents)
  {
  }

private:

  // The assignment operator is not implemented.
  MultiArrayOfCoefficientArraysTest&
  operator=(const MultiArrayOfCoefficientArraysTest&);

  //! @}
  //--------------------------------------------------------------------------
  //! \name Accessors and manipulators.
  //! @{
public:

  using Base::coefficientsData;

  //! @}
};

int
main()
{

  {
    const std::size_t Order = 1;
    const std::size_t Dimension = 1;
    typedef MultiArrayOfCoefficientArraysTest<double, Order, Dimension>
    GridArray;
    typedef GridArray::SizeList SizeList;
    typedef GridArray::Coefficients Coefficients;
    typedef container::MultiIndexRange<Dimension> Range;
    typedef container::MultiIndexRangeIterator<Dimension> Iterator;

    const std::size_t gridSize = 10;
    const std::size_t numberOfCells = gridSize - 1;
    const SizeList arrayExtents = {{20}};
    GridArray x(gridSize, arrayExtents);

    {
      GridArray y = x;
    }
    // Coefficients data manipulator.
    const Range range(arrayExtents);
    const Iterator begin = Iterator::begin(range);
    const Iterator end = Iterator::end(range);
    {
      for (Iterator i = begin; i != end; ++i) {
        Coefficients* g = x.coefficientsData(*i);
        const std::size_t offset = stlib::ext::sum(*i);
        for (std::size_t n = 0; n != numberOfCells; ++n) {
          std::fill(g[n].begin(), g[n].end(), offset * n);
        }
      }
    }
    // Coefficients accessor.
    {
      const GridArray& y = x;
      for (Iterator i = begin; i != end; ++i) {
        const Coefficients* g = y.coefficientsData(*i);
        const std::size_t offset = stlib::ext::sum(*i);
        for (std::size_t n = 0; n != numberOfCells; ++n) {
          for (std::size_t m = 0; m != g[n].size(); ++m) {
            assert(g[n][m] == offset * n);
          }
        }
      }
    }
  }

  {
    const std::size_t Order = 1;
    const std::size_t Dimension = 2;
    typedef MultiArrayOfCoefficientArraysTest<double, Order, Dimension>
    GridArray;
    typedef GridArray::SizeList SizeList;
    typedef GridArray::Coefficients Coefficients;
    typedef container::MultiIndexRange<Dimension> Range;
    typedef container::MultiIndexRangeIterator<Dimension> Iterator;

    const std::size_t gridSize = 10;
    const std::size_t numberOfCells = gridSize - 1;
    const SizeList arrayExtents = {{20, 30}};
    GridArray x(gridSize, arrayExtents);

    {
      GridArray y = x;
    }
    // Coefficients data manipulator.
    const Range range(arrayExtents);
    const Iterator begin = Iterator::begin(range);
    const Iterator end = Iterator::end(range);
    {
      for (Iterator i = begin; i != end; ++i) {
        Coefficients* g = x.coefficientsData(*i);
        const std::size_t offset = stlib::ext::sum(*i);
        for (std::size_t n = 0; n != numberOfCells; ++n) {
          std::fill(g[n].begin(), g[n].end(), offset * n);
        }
      }
    }
    // Coefficients accessor.
    {
      const GridArray& y = x;
      for (Iterator i = begin; i != end; ++i) {
        const Coefficients* g = y.coefficientsData(*i);
        const std::size_t offset = stlib::ext::sum(*i);
        for (std::size_t n = 0; n != numberOfCells; ++n) {
          for (std::size_t m = 0; m != g[n].size(); ++m) {
            assert(g[n][m] == offset * n);
          }
        }
      }
    }
  }

  {
    const std::size_t Order = 3;
    const std::size_t Dimension = 1;
    typedef MultiArrayOfCoefficientArraysTest<double, Order, Dimension>
    GridArray;
    typedef GridArray::SizeList SizeList;
    typedef GridArray::Coefficients Coefficients;
    typedef container::MultiIndexRange<Dimension> Range;
    typedef container::MultiIndexRangeIterator<Dimension> Iterator;

    const std::size_t gridSize = 10;
    const std::size_t numberOfCells = gridSize - 1;
    const SizeList arrayExtents = {{20}};
    GridArray x(gridSize, arrayExtents);

    {
      GridArray y = x;
    }
    // Coefficients data manipulator.
    const Range range(arrayExtents);
    const Iterator begin = Iterator::begin(range);
    const Iterator end = Iterator::end(range);
    {
      for (Iterator i = begin; i != end; ++i) {
        Coefficients* g = x.coefficientsData(*i);
        const std::size_t offset = stlib::ext::sum(*i);
        for (std::size_t n = 0; n != numberOfCells; ++n) {
          std::fill(g[n].begin(), g[n].end(), offset * n);
        }
      }
    }
    // Coefficients accessor.
    {
      const GridArray& y = x;
      for (Iterator i = begin; i != end; ++i) {
        const Coefficients* g = y.coefficientsData(*i);
        const std::size_t offset = stlib::ext::sum(*i);
        for (std::size_t n = 0; n != numberOfCells; ++n) {
          for (std::size_t m = 0; m != g[n].size(); ++m) {
            assert(g[n][m] == offset * n);
          }
        }
      }
    }
  }

  {
    const std::size_t Order = 3;
    const std::size_t Dimension = 2;
    typedef MultiArrayOfCoefficientArraysTest<double, Order, Dimension>
    GridArray;
    typedef GridArray::SizeList SizeList;
    typedef GridArray::Coefficients Coefficients;
    typedef container::MultiIndexRange<Dimension> Range;
    typedef container::MultiIndexRangeIterator<Dimension> Iterator;

    const std::size_t gridSize = 10;
    const std::size_t numberOfCells = gridSize - 1;
    const SizeList arrayExtents = {{20, 30}};
    GridArray x(gridSize, arrayExtents);

    {
      GridArray y = x;
    }
    // Coefficients data manipulator.
    const Range range(arrayExtents);
    const Iterator begin = Iterator::begin(range);
    const Iterator end = Iterator::end(range);
    {
      for (Iterator i = begin; i != end; ++i) {
        Coefficients* g = x.coefficientsData(*i);
        const std::size_t offset = stlib::ext::sum(*i);
        for (std::size_t n = 0; n != numberOfCells; ++n) {
          std::fill(g[n].begin(), g[n].end(), offset * n);
        }
      }
    }
    // Coefficients accessor.
    {
      const GridArray& y = x;
      for (Iterator i = begin; i != end; ++i) {
        const Coefficients* g = y.coefficientsData(*i);
        const std::size_t offset = stlib::ext::sum(*i);
        for (std::size_t n = 0; n != numberOfCells; ++n) {
          for (std::size_t m = 0; m != g[n].size(); ++m) {
            assert(g[n][m] == offset * n);
          }
        }
      }
    }
  }

  return 0;
}
