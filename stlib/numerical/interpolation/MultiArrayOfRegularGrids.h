// -*- C++ -*-

/*!
  \file numerical/interpolation/MultiArrayOfRegularGrids.h
  \brief A multi-dimensional array of regular grids.
*/

#if !defined(__numerical_interpolation_MultiArrayOfRegularGrids_h__)
#define __numerical_interpolation_MultiArrayOfRegularGrids_h__

#include "stlib/container/MultiArrayRef.h"

#include <vector>

namespace stlib
{
namespace numerical
{

//! A multi-dimensional array of regular grids.
/*!
  This class is intended to be used with
  numerical::MultiArrayOfRegularGridsReference.

  See the
  \ref interpolation_InterpolatingFunctionRegularGrid "interpolating functor"
  page for documentation on how to use this class.
*/
template<typename _T, std::size_t _GridDimension, std::size_t _ArrayDimension>
class MultiArrayOfRegularGrids
{
  //
  // Private types.
  //
private:

  typedef container::MultiIndexTypes<_ArrayDimension> ArrayMultiIndex;

  //
  // Public types.
  //
public:

  //! A reference to a grid.
  typedef container::MultiArrayRef<_T, _GridDimension> GridRef;
  //! A const reference to a grid.
  typedef container::MultiArrayConstRef<_T, _GridDimension> GridConstRef;

  //! The value type.
  typedef typename GridRef::value_type Value;
  //! The single index type.
  typedef typename GridRef::Index Index;

  //! The (multi) size type for the grids.
  typedef typename GridRef::SizeList GridSizeList;
  //! The (multi) index type for the grids.
  typedef typename GridRef::IndexList GridIndexList;

  //! The (multi) size type for the array.
  typedef typename ArrayMultiIndex::SizeList ArraySizeList;
  //! The (multi) index type for the array.
  typedef typename ArrayMultiIndex::IndexList ArrayIndexList;

  //
  // Data.
  //
private:

  //! A contiguous array of the grid data.
  std::vector<Value> _data;
  //! The multi-array extents.
  ArraySizeList _arrayExtents;
  //! The multi-array strides.
  ArrayIndexList _strides;
  //! Variable to reduce the cost of the grid manipulator.
  mutable GridRef _gridRef;
  //! Variable to reduce the cost of the grid accessor.
  mutable GridConstRef _gridConstRef;

  //--------------------------------------------------------------------------
  //! \name Constructors etc.
  //! @{
public:

  //! Construct from the grid extents and the multi-array extents.
  MultiArrayOfRegularGrids(const GridSizeList& gridExtents,
                           const ArraySizeList& arrayExtents) :
    _data(ext::product(gridExtents) * ext::product(arrayExtents)),
    _arrayExtents(arrayExtents),
    _strides(),
    _gridRef(0, gridExtents),
    _gridConstRef(0, gridExtents)
  {
    computeStrides();
  }

  //! Copy constructor.
  MultiArrayOfRegularGrids(const MultiArrayOfRegularGrids& other) :
    _data(other._data),
    _arrayExtents(other._arrayExtents),
    _strides(other._strides),
    _gridRef(0, other._gridRef.extents()),
    _gridConstRef(0, other._gridConstRef.extents())
  {
  }

  // The default destructor is fine.

private:

  // Assignment operator not implemented.
  const MultiArrayOfRegularGrids&
  operator=(const MultiArrayOfRegularGrids&);

  void
  computeStrides()
  {
    Index s = 1;
    for (std::size_t i = 0; i != _ArrayDimension; ++i) {
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
  const ArraySizeList&
  arrayExtents() const
  {
    return _arrayExtents;
  }

  //! Return a const pointer to the specified grid data.
  const Value*
  gridData(const ArrayIndexList& indices) const
  {
    return &_data[stlib::ext::dot(_strides, indices) * _gridConstRef.size()];
  }

  //! Return a const reference to the specified grid.
  const GridConstRef&
  grid(const ArrayIndexList& indices) const
  {
    _gridConstRef.setData(gridData(indices));
    return _gridConstRef;
  }

  //! @}
  //--------------------------------------------------------------------------
  //! \name Manipulators.
  //! @{
public:

  //! Return a pointer to the specified grid data.
  Value*
  gridData(const ArrayIndexList& indices)
  {
    return &_data[stlib::ext::dot(_strides, indices) * _gridRef.size()];
  }

  //! Return a reference to the specified grid.
  /*! When using this function you should copy the GridRef. */
  const GridRef&
  grid(const ArrayIndexList& indices)
  {
    _gridRef.setData(gridData(indices));
    return _gridRef;
  }

  //! @}
};

} // namespace numerical
}

#endif
