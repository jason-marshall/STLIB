// -*- C++ -*-

/*!
  \file numerical/interpolation/InterpolatingFunctionMultiArrayOf1DRegularGrids.h
  \brief Linear interpolation for a multi-dimensional %array of regular grids.
*/

#if !defined(__numerical_interpolation_InterpolatingFunctionMultiArrayOf1DRegularGrids_h__)
#define __numerical_interpolation_InterpolatingFunctionMultiArrayOf1DRegularGrids_h__

#include "stlib/numerical/interpolation/InterpolatingFunction1DRegularGridBase.h"
#include "stlib/numerical/interpolation/MultiArrayOf1DRegularGrids.h"

namespace stlib
{
namespace numerical
{


//! Linear interpolation for a multi-dimensional %array of regular grids.
/*!
  See the
  \ref interpolation_InterpolatingFunction1DRegularGrid "linear interpolation"
  page for documentation on how to use this class.
*/
template<typename _T, std::size_t _Order, std::size_t _Dimension>
class InterpolatingFunctionMultiArrayOf1DRegularGrids :
  public InterpolatingFunction1DRegularGridBase<_T, _Order>,
  public MultiArrayOf1DRegularGrids<_T, _Order, _Dimension>
{
  //
  // Private types.
  //
private:

  typedef InterpolatingFunction1DRegularGridBase<_T, _Order> FunctorBase;
  typedef MultiArrayOf1DRegularGrids<_T, _Order, _Dimension> DataBase;

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

  //! The (multi) size type for the array.
  typedef typename DataBase::SizeList SizeList;
  //! The (multi) index type for the array.
  typedef typename DataBase::IndexList IndexList;

  //! The single index type.
  typedef typename DataBase::Index Index;

  //--------------------------------------------------------------------------
  /*! \name Constructors etc.
    We use the default copy constructor and destructor. The assignment
    operator is not implemented.
  */
  //! @{
public:

  //! Construct from the grid size and the multi-array extents.
  /*!
    \param gridSize The grid size.
    \param lower The lower bound of the Cartesian domain.
    \param upper The upper bound of the Cartesian domain.
    \param arrayExtents The array extents.

    \note You must initialize the grid data before interpolating.
  */
  InterpolatingFunctionMultiArrayOf1DRegularGrids
  (const std::size_t gridSize, const double lower, const double upper,
   const SizeList& arrayExtents) :
    FunctorBase(gridSize, lower, upper),
    DataBase(gridSize, arrayExtents)
  {
  }

private:

  // The assignment operator is not implemented.
  const InterpolatingFunctionMultiArrayOf1DRegularGrids&
  operator=(const InterpolatingFunctionMultiArrayOf1DRegularGrids&);

  //! @}
  //--------------------------------------------------------------------------
  //! \name Interpolation.
  //! @{
public:

  //! Set the grid to use for the interpolation.
  void
  setGrid(const IndexList& indices)
  {
    FunctorBase::_grid = DataBase::_gridData(indices);
  }

  //! @}
};

} // namespace numerical
}

#endif
