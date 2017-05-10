// -*- C++ -*-

/*!
  \file numerical/polynomial/Polynomial.h
  \brief Polynomial with specified order.
*/

#if !defined(__numerical_Polynomial_h__)
#define __numerical_Polynomial_h__

#include <functional>

#include <array>

namespace stlib
{
namespace numerical
{


//-----------------------------------------------------------------------------
/*! \defgroup numerical_polynomial_static Evaluating Polynomials
  These functions evaluate polynomials whose order \e N is known at
  compile-time. For an %array of coefficients \e c, they evaluate
  \f$\sum_{n = 0}^{N} c_n x^n\f$.

  There are two functions that accept a random access constant iterator
  to the coefficients. For these functions one must explicitly specify
  the polynomial order \c N.  It cannot be deduced
  from the template parameters.  Below, we evaluate the quadratic polynomial
  <i>2 + 3 x + 5 x<sup>2</sup></i> at <i>x = 1.5</i> and then evaluate
  the polynomial and its derivative at <i>x = 2</i>.
  (Quadratic equations are so named because
  <em>quadratus</em> is Latin for "square.")
  \code
  const double c[] = {2, 3, 5};
  const double result = numerical::evaluatePolynomial<2>(c, 1.5);
  double derivative;
  const double value = numerical::evaluatePolynomial<2>(c, 2., &derivative);
  \endcode

  \warning Be careful of "off by one" errors.  The template parameter \c N
  is not the size of the coefficients array.  Its size is <tt>N + 1</tt>.

  There are two functions that accept a std::array of the coefficients.
  For this interface the order can be deduced from the size of the array.
  \code
  const std::array<double, 3> c = {{2, 3, 5}};
  const double result = numerical::evaluatePolynomial(c, 1.5);
  double derivative;
  const double value = numerical::evaluatePolynomial(c, 2., &derivative);
  \endcode
*/
//@{


//! Evaluate the polynomial with the specified coefficients.
template<std::size_t _Order, typename _RandomAccessIterator, typename _T>
_T
evaluatePolynomial(_RandomAccessIterator coefficients, _T x);


//! Evaluate the polynomial with the specified coefficients.
template<std::size_t _Size, typename _T>
inline
_T
evaluatePolynomial(const std::array<_T, _Size>& coefficients,
                   const _T x)
{
  return evaluatePolynomial < _Size - 1 > (coefficients.begin(), x);
}


//! Evaluate the polynomial value and derivative.
template<std::size_t _Order, typename _RandomAccessIterator, typename _T>
_T
evaluatePolynomial(_RandomAccessIterator coefficients, _T x, _T* derivative);


//! Evaluate the polynomial value and derivative.
template<std::size_t _Size, typename _T>
inline
_T
evaluatePolynomial(const std::array<_T, _Size>& coefficients,
                   const _T x, _T* derivative)
{
  return evaluatePolynomial < _Size - 1 > (coefficients.begin(), x, derivative);
}

//! Evaluate the polynomial value and first two derivatives.  Has not yet been tested.
template<std::size_t _Order, typename _RandomAccessIterator, typename _T>
_T
evaluatePolynomial(_RandomAccessIterator coefficients, _T x, _T* derivative,
                   _T* derivative2);

//! Evaluate the polynomial value and first two derivatives.
template<std::size_t _Size, typename _T>
inline
_T
evaluatePolynomial(const std::array<_T, _Size>& coefficients,
                   const _T x, _T* derivative, _T* derivative2)
{
  return evaluatePolynomial < _Size - 1 > (coefficients.begin(), x, derivative,
         derivative2);
}

//@}
//-----------------------------------------------------------------------------
/*! \defgroup numerical_polynomial_differentiating Differentiating Polynomials
  This function differentiates a polynomial whose order \e N is known at
  compile-time. The polynomial is specified with its coefficients \e c:
  \f$f(x) = \sum_{n = 0}^{N} c_n x^n\f$.



  The function accepts a random access constant iterator
  to the coefficients. One must explicitly specify
  the polynomial order \c N; it cannot be deduced
  from the template parameters.  Below, we differentiate the quadratic
  polynomial <i>2 + 3 x + 5 x<sup>2</sup></i> and then evaluate
  the derivative at <i>x = 2</i>.
  \code
  double c[] = {2, 3, 5};
  numerical::differentiatePolynomialCoefficients<2>(c);
  const double derivative = numerical::evaluatePolynomial<1>(c, 2.);
  \endcode

  \warning Be careful of "off by one" errors.  The template parameter \c N
  is not the size of the coefficients array.  Its size is <tt>N + 1</tt>.
*/
//@{


//! Differentiate the polynomial with the specified coefficients.
template<std::size_t _Order, typename _RandomAccessIterator>
void
differentiatePolynomialCoefficients(_RandomAccessIterator coefficients);

//@}

//! Polynomial with a specified order.
template < std::size_t _Order, typename _T = double >
class Polynomial :
  public std::unary_function<_T, _T>
{
  //
  // Private types.
  //
private:

  typedef std::unary_function<_T, _T> Base;

  //
  // Public types.
  //
public:

  //! The argument type.
  typedef typename Base::argument_type argument_type;
  //! The result type.
  typedef typename Base::result_type result_type;

  //
  // Member data.
  //
private:

  std::array < _T, _Order + 1 > _coefficients;

  //--------------------------------------------------------------------------
  //! \name Constructors etc.
  //! @{
public:

  //! Construct from the polynomial coefficients.
  template<typename _InputIterator>
  Polynomial(_InputIterator coefficients) :
    _coefficients()
  {
    setCoefficients(coefficients);
  }

  //! Construct from the polynomial coefficients.
  Polynomial(const std::array < _T, _Order + 1 > & coefficients) :
    _coefficients(coefficients)
  {
  }

  // The default copy constructor, assignment operator, and destructor are
  // fine.

  //! @}
  //--------------------------------------------------------------------------
  //! \name Manipulators.
  //! @{
public:

  //! Set the coefficients.
  template<typename _InputIterator>
  void
  setCoefficients(_InputIterator coefficients)
  {
    for (std::size_t i = 0; i != _coefficients.size(); ++i) {
      _coefficients[i] = *coefficients++;
    }
  }

  //! Set the coefficients.
  void
  setCoefficients(const std::array < _T, _Order + 1 > & coefficients)
  {
    _coefficients = coefficients;
  }

  //! @}
  //--------------------------------------------------------------------------
  //! \name Functor.
  //! @{
public:

  //! Evaluate the polynomial.
  result_type
  operator()(const argument_type x) const
  {
    return evaluatePolynomial(_coefficients, x);
  }

  //! Evaluate the polynomial value and derivative.
  result_type
  operator()(const argument_type x, result_type* derivative) const
  {
    return evaluatePolynomial(_coefficients, x, derivative);
  }

  //! @}
};


//! Convenience function for constructing a \c Polynomial.
/*! \relates Polynomial */
template<std::size_t _Size, typename _T>
inline
Polynomial < _Size - 1, _T >
constructPolynomial(const std::array<_T, _Size>& coefficients)
{
  Polynomial < _Size - 1, _T > p(coefficients);
  return p;
}


//! Convenience function for constructing a \c Polynomial.
/*! \relates Polynomial */
template<std::size_t _Order, typename _InputIterator>
inline
Polynomial<_Order, typename std::iterator_traits<_InputIterator>::value_type>
constructPolynomial(_InputIterator coefficients)
{
  Polynomial<_Order, typename std::iterator_traits<_InputIterator>::value_type>
  p(coefficients);
  return p;
}


} // namespace numerical
}

#define __numerical_polynomial_Polynomial_ipp__
#include "stlib/numerical/polynomial/Polynomial.ipp"
#undef __numerical_polynomial_Polynomial_ipp__

#endif
