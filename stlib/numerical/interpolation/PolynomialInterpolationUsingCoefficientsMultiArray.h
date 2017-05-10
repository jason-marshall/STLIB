// -*- C++ -*-

/*!
  \file numerical/interpolation/PolynomialInterpolationUsingCoefficientsMultiArray.h
  \brief Polynomial interpolation for a multi-dimensional %array of regular grids.
*/

#if !defined(__numerical_interpolation_PolynomialInterpolationUsingCoefficientsMultiArray_h__)
#define __numerical_interpolation_PolynomialInterpolationUsingCoefficientsMultiArray_h__

#include "stlib/numerical/interpolation/PolynomialInterpolationUsingCoefficientsBase.h"
#include "stlib/numerical/interpolation/MultiArrayOfCoefficientArrays.h"

namespace stlib
{
namespace numerical
{


//! Polynomial interpolation for a multi-dimensional %array of regular grids.
/*!
  \param _T The value type for the grid.
  \param _Order The interpolation order may be 1, 3, or 5.
  \param _Dimension The array dimension.

  See the
  \ref interpolation_PolynomialInterpolationUsingCoefficients "polynomial interpolation"
  page for documentation on how to use this class.
*/
template<typename _T, std::size_t _Order, std::size_t _Dimension>
class PolynomialInterpolationUsingCoefficientsMultiArray :
  public PolynomialInterpolationUsingCoefficientsBase<_T, _Order>,
  public MultiArrayOfCoefficientArrays<_T, _Order, _Dimension>
{
  //
  // Private types.
  //
private:

  typedef PolynomialInterpolationUsingCoefficientsBase<_T, _Order>
  FunctorBase;
  typedef MultiArrayOfCoefficientArrays<_T, _Order, _Dimension> DataBase;

  //
  // Public types.
  //
public:

  //! The (multi) size type for the array.
  typedef typename DataBase::SizeList SizeList;
  //! The (multi) index type for the array.
  typedef typename DataBase::IndexList IndexList;

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
  PolynomialInterpolationUsingCoefficientsMultiArray
  (const std::size_t gridSize, const double lower, const double upper,
   const SizeList& arrayExtents) :
    FunctorBase(gridSize, lower, upper),
    DataBase(gridSize, arrayExtents)
  {
  }

private:

  // The assignment operator is not implemented.
  const PolynomialInterpolationUsingCoefficientsMultiArray&
  operator=(const PolynomialInterpolationUsingCoefficientsMultiArray&);

  //! @}
  //--------------------------------------------------------------------------
  //! \name Interpolation.
  //! @{
public:

  //! Set the grid to use for the interpolation.
  void
  setGrid(const IndexList& indices)
  {
    FunctorBase::_coefficients = DataBase::coefficientsData(indices);
  }

  //! @}
};

} // namespace numerical
}

#endif
