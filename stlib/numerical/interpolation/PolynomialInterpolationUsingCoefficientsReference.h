// -*- C++ -*-

/*!
  \file numerical/interpolation/PolynomialInterpolationUsingCoefficientsReference.h
  \brief Polynomial interpolation on a 1-D regular grid.
*/

#if !defined(__numerical_interpolation_PolynomialInterpolationUsingCoefficientsReference_h__)
#define __numerical_interpolation_PolynomialInterpolationUsingCoefficientsReference_h__

#include "stlib/numerical/interpolation/PolynomialInterpolationUsingCoefficientsBase.h"

namespace stlib
{
namespace numerical
{

//! Polynomial interpolation on a 1-D regular grid.
/*!
  \param _T The value type for the grid.
  \param _Order The interpolation order may be 1, 3, or 5.
*/
template<typename _T, std::size_t _Order>
class PolynomialInterpolationUsingCoefficientsReference :
  public PolynomialInterpolationUsingCoefficientsBase<_T, _Order>
{
  //
  // Types.
  //
private:

  //! The functor type.
  typedef PolynomialInterpolationUsingCoefficientsBase<_T, _Order> Base;

public:

  //! The value type.
  typedef typename Base::Value Value;
  //! The %array of polynomial coefficients.
  typedef typename Base::Coefficients Coefficients;

  //
  // Using member data.
  //
protected:

  using Base::_coefficients;
  using Base::_numberOfCells;
  using Base::_lowerCorner;
  using Base::_inverseWidth;

  //--------------------------------------------------------------------------
  /*! \name Constructors etc.
    We use the default destructor. The memory for the grid is not deleted.
  */
  //! @{
public:

  //! Default constructor. Invalid data.
  PolynomialInterpolationUsingCoefficientsReference() :
    Base(1, 0, 1)
  {
  }

  //! Construct from the grid and the Cartesian domain.
  /*!
    \param coefficients Pointer to the coefficients data.
    \param numberOfCells The number of cells.
    \param lower The lower bound of the Cartesian domain.
    \param upper The upper bound of the Cartesian domain.

    The grid memory will be referenced, not copied.
  */
  PolynomialInterpolationUsingCoefficientsReference
  (Coefficients* coefficients, const std::size_t numberOfCells,
   const double lower, const double upper) :
    Base(numberOfCells + 1, lower, upper)
  {
    _coefficients = coefficients;
  }

  //! Copy constructor. Shallow copy. Reference the same memory.
  PolynomialInterpolationUsingCoefficientsReference
  (const PolynomialInterpolationUsingCoefficientsReference& other) :
    Base(other)
  {
  }

  //! Assignment operator. Deep copy. The grids must be the same size.
  PolynomialInterpolationUsingCoefficientsReference&
  operator=(const PolynomialInterpolationUsingCoefficientsReference& other)
  {
    if (this != &other) {
      Base::copy(other);
    }
    return *this;
  }

  //! @}
};

} // namespace numerical
}

#endif
