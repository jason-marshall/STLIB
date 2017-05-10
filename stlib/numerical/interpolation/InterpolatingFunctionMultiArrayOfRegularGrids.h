// -*- C++ -*-

/*!
  \file numerical/interpolation/InterpolatingFunctionMultiArrayOfRegularGrids.h
  \brief Interpolating function for a multi-dimensional %array of regular grids.
*/

#if !defined(__numerical_interpolation_InterpolatingFunctionMultiArrayOfRegularGrids_h__)
#define __numerical_interpolation_InterpolatingFunctionMultiArrayOfRegularGrids_h__

#include "stlib/numerical/interpolation/InterpolatingFunctionRegularGridBase.h"
#include "stlib/numerical/interpolation/MultiArrayOfRegularGrids.h"

#include "stlib/container/MultiArrayRef.h"

#include <vector>

namespace stlib
{
namespace numerical
{


//! Interpolating function for a multi-dimensional %array of regular grids.
/*!
  See the
  \ref interpolation_InterpolatingFunctionRegularGrid "interpolating functor"
  page for documentation on how to use this class.
*/
template<typename _T, std::size_t _GridDimension, std::size_t _ArrayDimension,
         std::size_t _DefaultOrder, bool _IsPeriodic = false>
class InterpolatingFunctionMultiArrayOfRegularGrids : public
  InterpolatingFunctionRegularGridBase<_T, _GridDimension,
  _DefaultOrder, _IsPeriodic>,
  MultiArrayOfRegularGrids<_T, _GridDimension, _ArrayDimension>
{
  //
  // Private types.
  //
private:

  typedef InterpolatingFunctionRegularGridBase<_T, _GridDimension,
          _DefaultOrder, _IsPeriodic>
          FunctorBase;
  typedef MultiArrayOfRegularGrids<_T, _GridDimension, _ArrayDimension>
  DataBase;

  //
  // Public types.
  //
public:

  //! The argument type is a Cartesian point.
  typedef typename FunctorBase::argument_type argument_type;
  //! The result type is the field.
  typedef typename FunctorBase::result_type result_type;
  //! The value type.
  typedef typename FunctorBase::Value Value;
  //! A Cartesian point.
  typedef typename FunctorBase::Point Point;
  //! A bounding box.
  typedef typename FunctorBase::BBox BBox;
  //! The array of function values.
  typedef typename FunctorBase::GridConstRef GridConstRef;

  //! A reference to a grid.
  typedef typename DataBase::GridRef GridRef;

  //! The (multi) size type for the grids.
  typedef typename DataBase::GridSizeList GridSizeList;
  //! The (multi) index type for the grids.
  typedef typename DataBase::GridIndexList GridIndexList;

  //! The (multi) size type for the array.
  typedef typename DataBase::ArraySizeList ArraySizeList;
  //! The (multi) index type for the array.
  typedef typename DataBase::ArrayIndexList ArrayIndexList;

  //! The single index type.
  typedef typename DataBase::Index Index;

  //
  // Member data.
  //
private:

  mutable GridConstRef _interpolationGrid;

  //--------------------------------------------------------------------------
  //! \name Constructors etc.
  //! @{
public:

  //! Construct from the grid extents and the multi-array extents.
  /*!
    \param gridExtents The grid extents.
    \param domain The Cartesian domain.
    \param arrayExtents The array extents.

    \note You must initialize the grid data before interpolating.
  */
  InterpolatingFunctionMultiArrayOfRegularGrids
  (const GridSizeList& gridExtents, const BBox& domain,
   const ArraySizeList& arrayExtents) :
    FunctorBase(gridExtents, domain),
    DataBase(gridExtents, arrayExtents),
    _interpolationGrid(0, gridExtents)
  {
  }

  //! Copy constructor.
  InterpolatingFunctionMultiArrayOfRegularGrids
  (const InterpolatingFunctionMultiArrayOfRegularGrids& other) :
    FunctorBase(other),
    DataBase(other),
    _interpolationGrid(0, other._interpolationGrid.extents())
  {
  }

  // The default destructor is fine.

private:

  // Assignment operator not implemented.
  const InterpolatingFunctionMultiArrayOfRegularGrids&
  operator=(const InterpolatingFunctionMultiArrayOfRegularGrids&);

  //! @}
  //--------------------------------------------------------------------------
  //! \name Interpolation.
  //! @{
public:

  using FunctorBase::interpolate;
  using FunctorBase::operator();
  using FunctorBase::snapToGrid;


  //! Set the grid to use for the interpolation.
  void
  setGrid(const ArrayIndexList& indices)
  {
    _interpolationGrid.setData(DataBase::gridData(indices));
    FunctorBase::initialize(&_interpolationGrid);
  }

  //! @}
  //--------------------------------------------------------------------------
  //! \name Accessors.
  //! @{
public:

  using DataBase::arrayExtents;
  using DataBase::gridData;
  using DataBase::grid;

  //! @}
  //--------------------------------------------------------------------------
  //! \name Manipulators.
  //! @{
public:

  //using DataBase::gridData;
  //using DataBase::grid;

  //! @}
};

} // namespace numerical
}

#endif
