// -*- C++ -*-

/*!
  \file numerical/interpolation/MultiArrayOf1DRegularGrids.h
  \brief A multi-dimensional array of 1-D regular grids for linear interpolation.
*/

#if !defined(__numerical_interpolation_MultiArrayOf1DRegularGrids_h__)
#define __numerical_interpolation_MultiArrayOf1DRegularGrids_h__

#include "stlib/container/MultiIndexTypes.h"

#include <vector>

namespace stlib
{
namespace numerical
{

//! A multi-dimensional array of 1-D regular grids for linear interpolation.
template<typename _T, std::size_t _Order, std::size_t _Dimension>
class MultiArrayOf1DRegularGrids
{
  //
  // Constants.
  //
private:

  BOOST_STATIC_CONSTEXPR std::size_t GhostWidth = _Order / 2;

  //
  // Types.
  //
private:

  typedef container::MultiIndexTypes<_Dimension> Types;

public:

  //! The value type.
  typedef _T Value;
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

  //! A contiguous array of the grid data.
  std::vector<Value> _data;
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
  MultiArrayOf1DRegularGrids
  (const std::size_t gridSize, const SizeList& arrayExtents) :
    _data((gridSize + 2 * GhostWidth) * ext::product(arrayExtents)),
    _gridSize(gridSize),
    _arrayExtents(arrayExtents),
    _strides()
  {
    static_assert(_Order == 1 || _Order == 3, "Not supported.");
    assert(! _data.empty());
    computeStrides();
  }

private:

  // The assignment operator is not implemented.
  MultiArrayOf1DRegularGrids&
  operator=(const MultiArrayOf1DRegularGrids&);

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

  //! Return a const pointer to the specified grid data.
  const Value*
  gridData(const IndexList& indices) const
  {
    return &_data[ext::dot(_strides, indices) * (_gridSize + 2 * GhostWidth)]
           + GhostWidth;
  }

  //! @}
  //--------------------------------------------------------------------------
  //! \name Manipulators.
  //! @{
protected:

  //! Return a pointer to the specified grid data.
  /*! This member function is protected for compatibility with the cubic
    interpolation functor. For that class the grid has ghost cells which must
    be set when the grid values are modified. Therefore write access to the
    grid is controlled.
    I use an underscore to avoid clashes with the accessor. I don't think that
    this should not be necessary, though. */
  Value*
  _gridData(const IndexList& indices)
  {
    return &_data[ext::dot(_strides, indices) * (_gridSize + 2 * GhostWidth)]
           + GhostWidth;
  }

  //! @}
};

} // namespace numerical
}

#endif
