// -*- C++ -*-

/*!
  \file numerical/interpolation/PolynomialInterpolationUsingCoefficientsVector.h
  \brief Polynomial interpolation for a vector of regular grids.
*/

#if !defined(__numerical_interpolation_PolynomialInterpolationUsingCoefficientsVector_h__)
#define __numerical_interpolation_PolynomialInterpolationUsingCoefficientsVector_h__

#include "stlib/numerical/interpolation/PolynomialInterpolationUsingCoefficientsBase.h"
#include "stlib/numerical/interpolation/VectorOfCoefficientArrays.h"

namespace stlib
{
namespace numerical
{


//! Polynomial interpolation for a vector of regular grids.
/*!
  \param _T The value type for the grid.
  \param _Order The interpolation order may be 1, 3, or 5.

  See the
  \ref interpolation_PolynomialInterpolationUsingCoefficients "polynomial interpolation"
  page for documentation on how to use this class.
*/
template<typename _T, std::size_t _Order>
class PolynomialInterpolationUsingCoefficientsVector :
  public PolynomialInterpolationUsingCoefficientsBase<_T, _Order>,
  public VectorOfCoefficientArrays<_T, _Order>
{
  //
  // Private types.
  //
private:

  typedef PolynomialInterpolationUsingCoefficientsBase<_T, _Order>
  FunctorBase;
  typedef VectorOfCoefficientArrays<_T, _Order> DataBase;

  //--------------------------------------------------------------------------
  /*! \name Constructors etc.
    We use the default copy constructor and destructor. The assignment
    operator is not implemented.
  */
  //! @{
public:

  //! Construct from the grid size and the number of grids.
  /*!
    \param gridSize The grid size.
    \param lower The lower bound of the Cartesian domain.
    \param upper The upper bound of the Cartesian domain.
    \param numberOfGrids The number of grids.

    \note You must initialize the grid data before interpolating.
  */
  PolynomialInterpolationUsingCoefficientsVector
  (const std::size_t gridSize, const double lower, const double upper,
   const std::size_t numberOfGrids) :
    FunctorBase(gridSize, lower, upper),
    DataBase(gridSize, numberOfGrids)
  {
  }

private:

  // The assignment operator is not implemented.
  const PolynomialInterpolationUsingCoefficientsVector&
  operator=(const PolynomialInterpolationUsingCoefficientsVector&);

  //! @}
  //--------------------------------------------------------------------------
  //! \name Interpolation.
  //! @{
public:

  //! Set the grid to use for the interpolation.
  void
  setGrid(const std::size_t i)
  {
    FunctorBase::_coefficients = DataBase::coefficientsData(i);
  }

  //! @}
};

} // namespace numerical
}

#endif
