// -*- C++ -*-

#if !defined(__numerical_polynomial_PolynomialGenericOrder_ipp__)
#error This file is an implementation detail of PolynomialGenericOrder.
#endif

namespace stlib
{
namespace numerical
{


template<typename _RandomAccessIterator, typename _T>
inline
_T
evaluatePolynomial(std::size_t order, _RandomAccessIterator coefficients,
                   const _T x)
{
  _T result = coefficients[order];
  while (order != 0) {
    result = x * result + coefficients[--order];
  }
  return result;
}


template<typename _RandomAccessIterator, typename _T>
inline
_T
evaluatePolynomial(std::size_t order, _RandomAccessIterator coefficients,
                   const _T x, _T* derivative)
{
  _T result = coefficients[order];
  *derivative = 0;
  while (order != 0) {
    *derivative = x** derivative + order * coefficients[order];
    --order;
    result = x * result + coefficients[order];
  }
  return result;
}


} // namespace numerical
}
