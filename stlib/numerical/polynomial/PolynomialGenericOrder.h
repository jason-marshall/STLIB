// -*- C++ -*-

/*!
  \file numerical/polynomial/PolynomialGenericOrder.h
  \brief PolynomialGenericOrder with specified order.
*/

#if !defined(__numerical_PolynomialGenericOrder_h__)
#define __numerical_PolynomialGenericOrder_h__

#include <functional>
#include <vector>

#include <cassert>

namespace stlib
{
namespace numerical
{


//-----------------------------------------------------------------------------
/*! \defgroup numerical_polynomial_generic Evaluating Polynomials with Generic Orders
  These functions evaluate polynomials with generic order \e N.
  For an %array of coefficients \e c, they evaluate
  \f$\sum_{n = 0}^{N} c_n x^n\f$.

  There are two functions that accept a random access constant iterator
  to the coefficients. For these functions one must specify
  the polynomial order as an argument.
  Below, we evaluate the quadratic polynomial
  <i>2 + 3 x + 5 x<sup>2</sup></i> at <i>x = 1.5</i> and then evaluate
  the polynomial and its derivative at <i>x = 2</i>.
  (Quadratic equations are so named because
  <em>quadratus</em> is Latin for "square.")
  \code
  const double c[] = {2, 3, 5};
  const double result = numerical::evaluatePolynomial(2, c, 1.5);
  double derivative;
  const double value = numerical::evaluatePolynomial(2, c, 2., &derivative);
  \endcode

  \warning Be careful of "off by one" errors.  The order \e N
  is not the size of the coefficients array.  Its size is <tt>N + 1</tt>.

  There are two functions that accept a std::vector of the coefficients.
  For this interface the order can be deduced from the size of the array.
  \code
  const double data[] = {2, 3, 5};
  const std::vector<double> c(data, data + sizeof(data) / sizeof(double));
  const double result = numerical::evaluatePolynomial(c, 1.5);
  double derivative;
  const double value = numerical::evaluatePolynomial(c, 2., &derivative);
  \endcode
*/
//@{


//! Evaluate the polynomial with the specified order and coefficients.
template<typename _RandomAccessIterator, typename _T>
_T
evaluatePolynomial(std::size_t order, _RandomAccessIterator coefficients, _T x);


//! Evaluate the polynomial with the specified coefficients.
template<typename _T>
inline
_T
evaluatePolynomial(const std::vector<_T>& coefficients, const _T x)
{
  return evaluatePolynomial(coefficients.size() - 1, coefficients.begin(), x);
}


//! Evaluate the polynomial value and derivative.
template<typename _RandomAccessIterator, typename _T>
_T
evaluatePolynomial(std::size_t order, _RandomAccessIterator coefficients, _T x,
                   _T* derivative);


//! Evaluate the polynomial value and derivative.
template<typename _T>
inline
_T
evaluatePolynomial(const std::vector<_T>& coefficients, const _T x,
                   _T* derivative)
{
  return evaluatePolynomial(coefficients.size() - 1, coefficients.begin(), x,
                            derivative);
}

//@}

//! Polynomial with a generic order.
template < typename _T = double >
class PolynomialGenericOrder :
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

  std::vector<_T> _coefficients;

  //--------------------------------------------------------------------------
  //! \name Constructors etc.
  //! @{
public:

  //! Construct from the polynomial coefficients.
  template<typename _InputIterator>
  PolynomialGenericOrder(_InputIterator begin, _InputIterator end) :
    _coefficients(begin, end)
  {
#ifdef STLIB_DEBUG
    assert(begin != end);
#endif
  }

  //! Construct from the polynomial order and coefficients.
  template<typename _InputIterator>
  PolynomialGenericOrder(const std::size_t order,
                         _InputIterator coefficients) :
    _coefficients(order + 1)
  {
    for (std::size_t i = 0; i != _coefficients.size(); ++i) {
      _coefficients[i] = *coefficients++;
    }
  }

  //! Construct from the polynomial coefficients.
  PolynomialGenericOrder(const std::vector<_T>& coefficients) :
    _coefficients(coefficients)
  {
#ifdef STLIB_DEBUG
    assert(! coefficients.empty());
#endif
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
  setCoefficients(_InputIterator begin, _InputIterator end)
  {
    _coefficients.clear();
#ifdef STLIB_DEBUG
    assert(begin != end);
#endif
    while (begin != end) {
      _coefficients.push_back(*begin++);
    }
  }

  //! Set the coefficients.
  template<typename _InputIterator>
  void
  setCoefficients(const std::size_t order, _InputIterator coefficients)
  {
    _coefficients.resize(order + 1);
    for (std::size_t i = 0; i != _coefficients.size(); ++i) {
      _coefficients[i] = *coefficients++;
    }
  }

  //! Set the coefficients.
  void
  setCoefficients(const std::vector<_T>& coefficients)
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


//! Convenience function for constructing a \c PolynomialGenericOrder.
/*! \relates PolynomialGenericOrder */
template<typename _T>
inline
PolynomialGenericOrder<_T>
constructPolynomialGenericOrder(const std::vector<_T>& coefficients)
{
  PolynomialGenericOrder<_T> p(coefficients);
  return p;
}


//! Convenience function for constructing a \c PolynomialGenericOrder.
/*! \relates PolynomialGenericOrder */
template<typename _InputIterator>
inline
PolynomialGenericOrder<typename std::iterator_traits<_InputIterator>::value_type>
constructPolynomialGenericOrder(const std::size_t order,
                                _InputIterator coefficients)
{
  PolynomialGenericOrder
  <typename std::iterator_traits<_InputIterator>::value_type>
  p(order, coefficients);
  return p;
}


} // namespace numerical
}

#define __numerical_polynomial_PolynomialGenericOrder_ipp__
#include "stlib/numerical/polynomial/PolynomialGenericOrder.ipp"
#undef __numerical_polynomial_PolynomialGenericOrder_ipp__

#endif
