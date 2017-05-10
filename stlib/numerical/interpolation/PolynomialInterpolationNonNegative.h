// -*- C++ -*-

/*!
  \file numerical/interpolation/PolynomialInterpolationNonNegative.h
  \brief Polynomial interpolation for non-negative, nearly singular data.
*/

#if !defined(__numerical_interpolation_PolynomialInterpolationNonNegative_h__)
#define __numerical_interpolation_PolynomialInterpolationNonNegative_h__

#include "stlib/numerical/interpolation/PolynomialInterpolationUsingCoefficients.h"

#include <cmath>

namespace stlib
{
namespace numerical
{

//! Polynomial interpolation for non-negative, nearly singular data.
/*!
  \param _T The value type for the grid.
  \param _Order The interpolation order may be 1, 3, or 5.

  The function must have non-negative values. This class is designed to handle
  functions than are nearly singular. That is, the function values may differ
  by many orders of magnitude within a single interpolation cell.

  To the input data we apply the transformation log(<em>x</em> + 1).
  The interpolation is performed on the transformed data using
  PolynomialInterpolationUsingCoefficients. After interpolation, we
  invert the transformation with the function exp(<em>x</em>) - 1.

  Note that polynomial interpolation is ill-suited for modelling
  nearly singular behavior. This class essentially regularizes singularities
  before performing the interpolation. This regularization comes at a cost.
  Check out the file Transformed.nb to see the distortion introduced by
  the transformation.
*/
template<typename _T, std::size_t _Order>
class PolynomialInterpolationNonNegative :
  public PolynomialInterpolationUsingCoefficients<_T, _Order>
{
  //
  // Types.
  //
private:

  //! The base does the interpolation.
  typedef PolynomialInterpolationUsingCoefficients<_T, _Order> Base;

public:

  //! The value type.
  typedef typename Base::Value Value;

  //--------------------------------------------------------------------------
  /*! \name Constructors etc.
    We use the synthesized copy constructor, assignment operator, and
    destructor.
  */
  //! @{
public:

  //! Default constructor. Invalid data.
  PolynomialInterpolationNonNegative() :
    Base()
  {
  }

  //! Construct from the grid size and the Cartesian domain.
  /*!
    \param gridSize The number of grid points.
    \param lower The lower bound of the Cartesian domain.
    \param upper The upper bound of the Cartesian domain.

    The grid values are not initialized.
  */
  PolynomialInterpolationNonNegative(const std::size_t gridSize,
                                     const double lower, const double upper) :
    Base(gridSize, lower, upper)
  {
  }

  //! Construct from the grid and the Cartesian domain.
  /*!
    \param f The first in the range of function values.
    \param gridSize The number of grid points.
    \param lower The lower bound of the Cartesian domain.
    \param upper The upper bound of the Cartesian domain.
  */
  template<typename _ForwardIterator>
  PolynomialInterpolationNonNegative(_ForwardIterator f,
                                     const std::size_t gridSize,
                                     const double lower, const double upper) :
    Base(gridSize, lower, upper)
  {
    setGridValues(f);
  }

#if 0
  // CONTINUE
  //! Construct from the grid and the Cartesian domain.
  /*!
    \param f The first in the range of function values.
    \param df The first in the range of first derivative values.
    \param gridSize The number of grid points.
    \param lower The lower bound of the Cartesian domain.
    \param upper The upper bound of the Cartesian domain.
  */
  template<typename _ForwardIterator>
  PolynomialInterpolationNonNegative(_ForwardIterator f,
                                     _ForwardIterator df,
                                     const std::size_t gridSize,
                                     const double lower,
                                     const double upper) :
    Base(gridSize, lower, upper)
  {
    allocate();
    Base::setGridValues(f, df);
  }

  //! Construct from the grid and the Cartesian domain.
  /*!
    \param f The first in the range of function values.
    \param df The first in the range of first derivative values.
    \param ddf The first in the range of second derivative values.
    \param gridSize The number of grid points.
    \param lower The lower bound of the Cartesian domain.
    \param upper The upper bound of the Cartesian domain.
  */
  template<typename _ForwardIterator>
  PolynomialInterpolationNonNegative(_ForwardIterator f,
                                     _ForwardIterator df,
                                     _ForwardIterator ddf,
                                     const std::size_t gridSize,
                                     const double lower,
                                     const double upper) :
    Base(gridSize, lower, upper)
  {
    allocate();
    Base::setGridValues(f, df, ddf);
  }
#endif

  //! @}
  //--------------------------------------------------------------------------
  //! \name Interpolation.
  //! @{
public:

  //! Interpolate the field at the specified point.
  typename Base::result_type
  operator()(typename Base::argument_type x) const
  {
    return inverseTransform(Base::operator()(x));
  }

#if 0
  // CONTINUE
  //! Interpolate the function and its derivative.
  result_type
  operator()(argument_type x, Value* derivative) const;

  //! Interpolate the function and its first and second derivatives.
  result_type
  operator()(argument_type x, Value* firstDerivative, Value* secondDerivative)
  const;

  //! Interpolate the field at the specified point.
  /*! The interpolation order must be specified explicitly and must be
    no greater than the default order used to construct this class. */
  template<std::size_t _InterpolationOrder>
  result_type
  interpolate(argument_type x) const;

  //! Interpolate the function and its derivative.
  /*! The interpolation order must be specified explicitly and must be
    no greater than the default order used to construct this class. */
  template<std::size_t _InterpolationOrder>
  result_type
  interpolate(argument_type x, Value* derivative) const;

  //! Interpolate the function and its first and second derivatives.
  /*! The interpolation order must be specified explicitly and must be
    no greater than the default order used to construct this class. */
  template<std::size_t _InterpolationOrder>
  result_type
  interpolate(argument_type x, Value* firstDerivative, Value* secondDerivative)
  const;
#endif

  //! Use the grid values to set the polynomial coefficient values.
  /*!
    \param f The first in the range of function values.
    This overrides the member function from
    PolynomialInterpolationUsingCoefficientsBase.
  */
  template<typename _ForwardIterator>
  void
  setGridValues(_ForwardIterator f)
  {
    // Transform the function values with the logarithm.
    std::vector<Value> transformed(Base::getNumberOfGridPoints());
    for (std::size_t i = 0; i != transformed.size(); ++i) {
      transformed[i] = transform(*f++);
    }
    Base::setGridValues(transformed.begin());
  }

protected:

  Value
  transform(const Value x) const
  {
    assert(x >= 0);
    return std::log(x + 1);
  }

  Value
  inverseTransform(const Value x) const
  {
    return std::exp(x) - 1;
  }

  //! @}

};

} // namespace numerical
}

#endif
