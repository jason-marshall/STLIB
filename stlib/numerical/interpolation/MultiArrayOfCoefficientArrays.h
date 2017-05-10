// -*- C++ -*-

/*!
  \file numerical/interpolation/MultiArrayOfCoefficientArrays.h
  \brief A multi-dimensional array of coefficient arrays for polynomial interpolation.
*/

#if !defined(__numerical_interpolation_MultiArrayOfCoefficientArrays_h__)
#define __numerical_interpolation_MultiArrayOfCoefficientArrays_h__

#include "stlib/container/MultiIndexTypes.h"

#include <vector>

namespace stlib
{
namespace numerical
{

//! A multi-dimensional array of coefficient arrays for polynomial interpolation.
template<typename _T, std::size_t _Order, std::size_t _Dimension>
class MultiArrayOfCoefficientArrays
{
  //
  // Types.
  //
private:

  typedef container::MultiIndexTypes<_Dimension> Types;

public:

  //! The polynomial coefficients.
  typedef std::array < _T, _Order + 1 > Coefficients;
  //! The single index type.
  typedef typename Types::Index Index;
  //! The (multi) size type for the array.
  typedef typename Types::SizeList SizeList;
  //! The (multi) index type for the array.
  typedef typename Types::IndexList IndexList;

  //
  // Data.
  //
private:

  //! A contiguous array of the coefficients.
  std::vector<Coefficients> _data;
  //! The number of elements in a grid.
  const std::size_t _gridSize;
  //! The multi-array extents.
  const SizeList _arrayExtents;
  //! The multi-array strides.
  IndexList _strides;

  //--------------------------------------------------------------------------
  /*! \name Constructors etc.
    We use the default copy constructor and destructor. The assignment
    operator is not implemented.
  */
  //! @{
public:

  //! Construct from the grid size and the multi-array extents.
  MultiArrayOfCoefficientArrays
  (const std::size_t gridSize, const SizeList& arrayExtents) :
    _data((gridSize - 1) * ext::product(arrayExtents)),
    _gridSize(gridSize),
    _arrayExtents(arrayExtents),
    _strides()
  {
    static_assert(_Order == 1 || _Order == 3 || _Order == 5, "Not supported.");
    assert(! _data.empty());
    computeStrides();
  }

private:

  // The assignment operator is not implemented.
  MultiArrayOfCoefficientArrays&
  operator=(const MultiArrayOfCoefficientArrays&);

  //! Compute the strides used for array indexing.
  void
  computeStrides()
  {
    Index s = 1;
    for (std::size_t i = 0; i != _Dimension; ++i) {
      _strides[i] = s;
      s *= _arrayExtents[i];
    }
  }

  //! @}
  //--------------------------------------------------------------------------
  //! \name Accessors.
  //! @{
public:

  //! Return the array extents.
  const SizeList&
  arrayExtents() const
  {
    return _arrayExtents;
  }

protected:

  //! Return a const pointer to the specified coefficients data.
  const Coefficients*
  coefficientsData(const IndexList& indices) const
  {
    return &_data[ext::dot(_strides, indices) * (_gridSize  - 1)];
  }

  //! @}
  //--------------------------------------------------------------------------
  //! \name Manipulators.
  //! @{
protected:

  //! Return a pointer to the specified coefficients data.
  Coefficients*
  coefficientsData(const IndexList& indices)
  {
    return &_data[ext::dot(_strides, indices) * (_gridSize  - 1)];
  }

  //! @}
};

} // namespace numerical
}

#endif
