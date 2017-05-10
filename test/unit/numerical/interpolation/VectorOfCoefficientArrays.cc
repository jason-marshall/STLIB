// -*- C++ -*-

#include "stlib/numerical/interpolation/VectorOfCoefficientArrays.h"

#include <cassert>

using namespace stlib;

template<typename _T, std::size_t _Order>
class VectorOfCoefficientArraysTest :
  public numerical::VectorOfCoefficientArrays<_T, _Order>
{
  //
  // Types.
  //
private:

  typedef numerical::VectorOfCoefficientArrays<_T, _Order> Base;

  //--------------------------------------------------------------------------
  /*! \name Constructors etc.
    We use the default copy constructor and destructor. The assignment
    operator is not implemented.
  */
  //! @{
public:

  //! Construct from the grid size and the multi-array extents.
  VectorOfCoefficientArraysTest(const std::size_t gridSize,
                                const std::size_t numberOfGrids) :
    Base(gridSize, numberOfGrids)
  {
  }

private:

  // The assignment operator is not implemented.
  VectorOfCoefficientArraysTest&
  operator=(const VectorOfCoefficientArraysTest&);

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
    typedef VectorOfCoefficientArraysTest<double, Order> GridVector;
    typedef GridVector::Coefficients Coefficients;

    const std::size_t gridSize = 10;
    const std::size_t numberOfCells = gridSize - 1;
    const std::size_t numberOfGrids = 20;
    GridVector x(gridSize, numberOfGrids);

    {
      GridVector y = x;
    }
    // Coefficients data manipulator.
    for (std::size_t i = 0; i != numberOfGrids; ++i) {
      Coefficients* g = x.coefficientsData(i);
      const std::size_t offset = i;
      for (std::size_t n = 0; n != numberOfCells; ++n) {
        std::fill(g[n].begin(), g[n].end(), offset * n);
      }
    }
    // Coefficients accessor.
    {
      const GridVector& y = x;
      for (std::size_t i = 0; i != numberOfGrids; ++i) {
        const Coefficients* g = y.coefficientsData(i);
        const std::size_t offset = i;
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
    typedef VectorOfCoefficientArraysTest<double, Order> GridVector;
    typedef GridVector::Coefficients Coefficients;

    const std::size_t gridSize = 10;
    const std::size_t numberOfCells = gridSize - 1;
    const std::size_t numberOfGrids = 20;
    GridVector x(gridSize, numberOfGrids);

    {
      GridVector y = x;
    }
    // Coefficients data manipulator.
    for (std::size_t i = 0; i != numberOfGrids; ++i) {
      Coefficients* g = x.coefficientsData(i);
      const std::size_t offset = i;
      for (std::size_t n = 0; n != numberOfCells; ++n) {
        std::fill(g[n].begin(), g[n].end(), offset * n);
      }
    }
    // Coefficients accessor.
    {
      const GridVector& y = x;
      for (std::size_t i = 0; i != numberOfGrids; ++i) {
        const Coefficients* g = y.coefficientsData(i);
        const std::size_t offset = i;
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
