// -*- C++ -*-

/*!
  \file numerical/interpolation/VectorOfCoefficientArrays.h
  \brief A vector of coefficient arrays for polynomial interpolation.
*/

#if !defined(__numerical_interpolation_VectorOfCoefficientArrays_h__)
#define __numerical_interpolation_VectorOfCoefficientArrays_h__

#include <vector>

#include <array>

#include <cassert>

namespace stlib
{
namespace numerical
{


//! A vector of coefficient arrays for polynomial interpolation.
template<typename _T, std::size_t _Order>
class VectorOfCoefficientArrays
{
public:

  //! The polynomial coefficients.
  typedef std::array < _T, _Order + 1 > Coefficients;

  //
  // Data.
  //
private:

  //! A contiguous array of the coefficients.
  std::vector<Coefficients> _data;
  //! The number of elements in a grid.
  const std::size_t _gridSize;
  //! The number of grids.
  const std::size_t _numberOfGrids;

  //--------------------------------------------------------------------------
  /*! \name Constructors etc.
    We use the default copy constructor and destructor. The assignment
    operator is not implemented.
  */
  //! @{
public:

  //! Construct from the grid size and the multi-array extents.
  VectorOfCoefficientArrays(const std::size_t gridSize,
                            const std::size_t numberOfGrids) :
    _data((gridSize - 1) * numberOfGrids),
    _gridSize(gridSize),
    _numberOfGrids(numberOfGrids)
  {
    static_assert(_Order == 1 || _Order == 3 || _Order == 5, "Not supported.");
    assert(! _data.empty());
  }

private:

  // The assignment operator is not implemented.
  VectorOfCoefficientArrays&
  operator=(const VectorOfCoefficientArrays&);

  //! @}
  //--------------------------------------------------------------------------
  //! \name Accessors.
  //! @{
public:

  //! Return the number of grids.
  std::size_t
  getNumberOfGrids() const
  {
    return _numberOfGrids;
  }

protected:

  //! Return a const pointer to the specified coefficients data.
  const Coefficients*
  coefficientsData(const std::size_t& i) const
  {
    return &_data[i * (_gridSize  - 1)];
  }

  //! @}
  //--------------------------------------------------------------------------
  //! \name Manipulators.
  //! @{
protected:

  //! Return a pointer to the specified coefficients data.
  Coefficients*
  coefficientsData(const std::size_t& i)
  {
    return &_data[i * (_gridSize  - 1)];
  }

  //! @}
};

} // namespace numerical
}

#endif
