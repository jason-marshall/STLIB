// -*- C++ -*-

#if !defined(__numerical_polynomial_Polynomial_ipp__)
#error This file is an implementation detail of Polynomial.
#endif

namespace stlib
{
namespace numerical
{


template<std::size_t _Order, typename _RandomAccessIterator, typename _T>
inline
_T
evaluatePolynomial(_RandomAccessIterator coefficients, const _T x)
{
  _T result = coefficients[_Order];
  std::size_t n = _Order;
  while (n != 0) {
    result = x * result + coefficients[--n];
  }
  return result;
}


template<std::size_t _Order, typename _RandomAccessIterator, typename _T>
inline
_T
evaluatePolynomial(_RandomAccessIterator coefficients, const _T x,
                   _T* derivative)
{
  _T result = coefficients[_Order];
  *derivative = 0;
  std::size_t n = _Order;
  while (n != 0) {
    *derivative = x** derivative + n * coefficients[n];
    --n;
    result = x * result + coefficients[n];
  }
  return result;
}


template<std::size_t _Order, typename _RandomAccessIterator, typename _T>
inline
_T
evaluatePolynomial(_RandomAccessIterator coefficients, const _T x,
                   _T* derivative, _T* derivative2)
{
  _T result = coefficients[_Order];
  *derivative = 0;
  *derivative2 = 0;
  std::size_t n = _Order;
  while (n != 1) {
    *derivative2 = x** derivative2 + n * (n - 1) * coefficients[n];
    *derivative = x** derivative + n * coefficients[n];
    --n;
    result = x * result + coefficients[n];
  }
  if (n == 1) {
    *derivative = x** derivative + coefficients[1];
    result = x * result + coefficients[0];
  }
  return result;
}


// Differentiate the polynomial with the specified coefficients.
template<std::size_t _Order, typename _RandomAccessIterator>
inline
void
differentiatePolynomialCoefficients(_RandomAccessIterator coefficients)
{
  for (std::size_t i = 1; i < _Order + 1; ++i) {
    coefficients[i - 1] = i * coefficients[i];
  }
  coefficients[_Order] = 0;
}


} // namespace numerical
}
