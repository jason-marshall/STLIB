// -*- C++ -*-

/*!
  \file numerical/interpolation/PolynomialInterpolationUsingCoefficients.h
  \brief Polynomial interpolation on a 1-D regular grid.
*/

#if !defined(__numerical_interpolation_PolynomialInterpolationUsingCoefficients_h__)
#define __numerical_interpolation_PolynomialInterpolationUsingCoefficients_h__

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
class PolynomialInterpolationUsingCoefficients :
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

  //
  // Using member data.
  //
protected:

  using Base::_coefficients;
  using Base::_numberOfCells;
  using Base::_lowerCorner;
  using Base::_inverseWidth;

  //--------------------------------------------------------------------------
  //! \name Constructors etc.
  //! @{
public:

  //! Default constructor. Invalid data.
  PolynomialInterpolationUsingCoefficients() :
    Base(1, 0, 1)
  {
  }

  //! Construct from the grid size and the Cartesian domain.
  /*!
    \param gridSize The number of grid points.
    \param lower The lower bound of the Cartesian domain.
    \param upper The upper bound of the Cartesian domain.

    The grid values are not initialized.
  */
  PolynomialInterpolationUsingCoefficients(const std::size_t gridSize,
      double lower, double upper) :
    Base(gridSize, lower, upper)
  {
    // There must be at least one cell.
    assert(gridSize >= 2);
    allocate();
  }

  //! Construct from the grid and the Cartesian domain.
  /*!
    \param f The first in the range of function values.
    \param gridSize The number of grid points.
    \param lower The lower bound of the Cartesian domain.
    \param upper The upper bound of the Cartesian domain.
  */
  template<typename _ForwardIterator>
  PolynomialInterpolationUsingCoefficients(_ForwardIterator f,
      const std::size_t gridSize,
      double lower, double upper) :
    Base(gridSize, lower, upper)
  {
    allocate();
    Base::setGridValues(f);
  }

  //! Construct from the grid and the Cartesian domain.
  /*!
    \param f The first in the range of function values.
    \param df The first in the range of first derivative values.
    \param gridSize The number of grid points.
    \param lower The lower bound of the Cartesian domain.
    \param upper The upper bound of the Cartesian domain.
  */
  template<typename _ForwardIterator>
  PolynomialInterpolationUsingCoefficients(_ForwardIterator f,
      _ForwardIterator df,
      const std::size_t gridSize,
      double lower, double upper) :
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
  PolynomialInterpolationUsingCoefficients(_ForwardIterator f,
      _ForwardIterator df,
      _ForwardIterator ddf,
      const std::size_t gridSize,
      double lower, double upper) :
    Base(gridSize, lower, upper)
  {
    allocate();
    Base::setGridValues(f, df, ddf);
  }

  //! Copy constructor.
  PolynomialInterpolationUsingCoefficients
  (const PolynomialInterpolationUsingCoefficients& other) :
    Base(other)
  {
    allocate();
    Base::copyCoefficients(other);
  }

  //! Assignment operator.
  PolynomialInterpolationUsingCoefficients&
  operator=(const PolynomialInterpolationUsingCoefficients& other)
  {
    if (this != &other) {
      // Re-allocate memory if necessary.
      if (other._numberOfCells != _numberOfCells) {
        destroy();
        _numberOfCells = other._numberOfCells;
        allocate();
      }
      // Copy the grid values and other member variables.
      Base::copy(other);
    }
    return *this;
  }

  ~PolynomialInterpolationUsingCoefficients()
  {
    destroy();
  }

private:

  void
  allocate()
  {
    if (_numberOfCells > 0) {
      _coefficients = new typename Base::Coefficients[_numberOfCells];
    }
    else {
      _coefficients = 0;
    }
  }

  void
  destroy()
  {
    if (_coefficients) {
      delete[] _coefficients;
    }
  }

  //! @}
};

} // namespace numerical
}

#endif
