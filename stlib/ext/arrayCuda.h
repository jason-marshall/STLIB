// -*- C++ -*-

/*!
  \file
  \brief Functions for array.
*/

#if !defined(__ext_arrayCuda_h__)
#define __ext_arrayCuda_h__

#include "stlib/cuda/array.h"

#ifndef __CUDA_ARCH__
#include <iostream>
#endif

namespace std
{
namespace tr1
{

template<typename _T, std::size_t N>
inline
__device__
__host__
bool
operator==(const array<_T, N>& a, const array<_T, N>& b)
{
  for (std::size_t i = 0; i != N; ++i) {
    if (a[i] != b[i]) {
      return false;
    }
  }
  return true;
}

template<typename _T, std::size_t N>
inline
__device__
__host__
bool
operator!=(const array<_T, N>& a, const array<_T, N>& b)
{
  return !(a == b);
}

template<typename _T, std::size_t N>
inline
__device__
__host__
bool
operator<(const array<_T, N>& a, const array<_T, N>& b)
{
  for (std::size_t i = 0; i != N; ++i) {
    if (a[i] < b[i]) {
      return true;
    }
    else if (a[i] > b[i]) {
      break;
    }
  }
  return false;
}

template<typename _T, std::size_t N>
inline
__device__
__host__
bool
operator>(const array<_T, N>& a, const array<_T, N>& b)
{
  return b < a;
}

template<typename _T, std::size_t N>
inline
__device__
__host__
bool
operator<=(const array<_T, N>& a, const array<_T, N>& b)
{
  return !(a > b);
}

template<typename _T, std::size_t N>
inline
__device__
__host__
bool
operator>=(const array<_T, N>& a, const array<_T, N>& b)
{
  return !(a < b);
}

template<typename _T, std::size_t N>
inline
__device__
__host__
void
swap(array<_T, N>& a, array<_T, N>& b)
{
  a.swap(b);
}

// Note: The tuple interface is not supported.

/*!
\page cudaArray Extensions to cuda::array

Here we provide functions to extend the functionality of the cuda::array
class. The functions are grouped into
the following categories:
- \ref cudaArrayAssignmentScalar
- \ref cudaArrayAssignmentArray
- \ref cudaArrayUnary
- \ref cudaArrayBinary
- \ref cudaArrayFile
- \ref cudaArrayMathematical
- \ref cudaArrayMake
*/

//-------------------------------------------------------------------------
/*! \defgroup cudaArrayAssignmentScalar Assignment Operators with a Scalar Operand.

  These functions apply an assignment operation to each element of the array.
  Thus you can add, subtract, etc. a scalar from each element.
  \verbatim
  cuda::array<double, 3> x = {{0, 0, 0}};
  x += 2;
  x -= 3;
  x *= 5;
  x /= 7; \endverbatim

  \verbatim
  cuda::array<unsigned, 3> x = {{0, 0, 0}};
  x %= 2;
  x <<= 3;
  x >>= 5; \endverbatim
*/
//@{

//! Array-scalar addition.
template<typename _T, std::size_t N, typename _T2>
inline
__device__
__host__
array<_T, N>&
operator+=(array<_T, N>& x, const _T2& value)
{
  for (std::size_t i = 0; i != N; ++i) {
    x[i] += value;
  }
  return x;
}

//! Array-scalar subtraction.
template<typename _T, std::size_t N, typename _T2>
inline
__device__
__host__
array<_T, N>&
operator-=(array<_T, N>& x, const _T2& value)
{
  for (std::size_t i = 0; i != N; ++i) {
    x[i] -= value;
  }
  return x;
}

//! Array-scalar multiplication.
template<typename _T, std::size_t N, typename _T2>
inline
__device__
__host__
array<_T, N>&
operator*=(array<_T, N>& x, const _T2& value)
{
  for (std::size_t i = 0; i != N; ++i) {
    x[i] *= value;
  }
  return x;
}

//! Array-scalar division.
template<typename _T, std::size_t N, typename _T2>
inline
__device__
__host__
array<_T, N>&
operator/=(array<_T, N>& x, const _T2& value)
{
#if defined(STLIB_DEBUG) && ! defined(__CUDA_ARCH__)
  assert(value != 0);
#endif
  for (std::size_t i = 0; i != N; ++i) {
    x[i] /= value;
  }
  return x;
}

//! Array-scalar modulus.
template<typename _T, std::size_t N, typename _T2>
inline
__device__
__host__
array<_T, N>&
operator%=(array<_T, N>& x, const _T2& value)
{
#if defined(STLIB_DEBUG) && ! defined(__CUDA_ARCH__)
  assert(value != 0);
#endif
  for (std::size_t i = 0; i != N; ++i) {
    x[i] %= value;
  }
  return x;
}

//! Left shift.
template<typename _T, std::size_t N>
inline
__device__
__host__
array<_T, N>&
operator<<=(array<_T, N>& x, const int offset)
{
  for (std::size_t i = 0; i != N; ++i) {
    x[i] <<= offset;
  }
  return x;
}

//! Right shift.
template<typename _T, std::size_t N>
inline
__device__
__host__
array<_T, N>&
operator>>=(array<_T, N>& x, const int offset)
{
  for (std::size_t i = 0; i != N; ++i) {
    x[i] >>= offset;
  }
  return x;
}

//@}
//-------------------------------------------------------------------------
/*! \defgroup cudaArrayAssignmentArray Assignment Operators with an Array Operand.

  These functions define assignment operations between arrays. The assignment
  operation is applied element-wise.
  \verbatim
  cuda::array<double, 3> x, y;
  ...
  x += y;
  x -= y;
  x *= y;
  x /= y; \endverbatim
  \verbatim
  cuda::array<unsigned, 3> x, y;
  ...
  x %= y;
  x <<= y;
  x >>= y; \endverbatim
*/
//@{

//! Array-array addition.
template<typename _T, std::size_t N, typename _T2>
inline
__device__
__host__
array<_T, N>&
operator+=(array<_T, N>& x, const array<_T2, N>& y)
{
  for (std::size_t n = 0; n != x.size(); ++n) {
    x[n] += y[n];
  }
  return x;
}

//! Array-array subtraction.
template<typename _T, std::size_t N, typename _T2>
inline
__device__
__host__
array<_T, N>&
operator-=(array<_T, N>& x, const array<_T2, N>& y)
{
  for (std::size_t n = 0; n != x.size(); ++n) {
    x[n] -= y[n];
  }
  return x;
}

//! Array-array multiplication.
template<typename _T, std::size_t N, typename _T2>
inline
__device__
__host__
array<_T, N>&
operator*=(array<_T, N>& x, const array<_T2, N>& y)
{
  for (std::size_t n = 0; n != x.size(); ++n) {
    x[n] *= y[n];
  }
  return x;
}

//! Array-array division.
template<typename _T, std::size_t N, typename _T2>
inline
__device__
__host__
array<_T, N>&
operator/=(array<_T, N>& x, const array<_T2, N>& y)
{
  for (std::size_t n = 0; n != x.size(); ++n) {
#if defined(STLIB_DEBUG) && ! defined(__CUDA_ARCH__)
    assert(y[n] != 0);
#endif
    x[n] /= y[n];
  }
  return x;
}

//! Array-array modulus.
template<typename _T, std::size_t N, typename _T2>
inline
__device__
__host__
array<_T, N>&
operator%=(array<_T, N>& x, const array<_T2, N>& y)
{
  for (std::size_t n = 0; n != x.size(); ++n) {
#if defined(STLIB_DEBUG) && ! defined(__CUDA_ARCH__)
    assert(y[n] != 0);
#endif
    x[n] %= y[n];
  }
  return x;
}

//! Array-array left shift.
template<typename _T1, std::size_t N, typename _T2>
inline
__device__
__host__
array<_T1, N>&
operator<<=(array<_T1, N>& x, const array<_T2, N>& y)
{
  for (std::size_t n = 0; n != x.size(); ++n) {
    x[n] <<= y[n];
  }
  return x;
}

//! Array-array right shift.
template<typename _T1, std::size_t N, typename _T2>
inline
__device__
__host__
array<_T1, N>&
operator>>=(array<_T1, N>& x, const array<_T2, N>& y)
{
  for (std::size_t n = 0; n != x.size(); ++n) {
    x[n] >>= y[n];
  }
  return x;
}

//@}
//-------------------------------------------------------------------------
/*! \defgroup cudaArrayUnary Unary Operators

  These functions define unary operations for arrays.
  \verbatim
  cuda::array<double, 3> x, y;
  ...
  x = +y;
  x = -y; \endverbatim
*/
//@{

//! Unary positive operator.
template<typename _T, std::size_t N>
inline
__device__
__host__
const array<_T, N>&
operator+(const array<_T, N>& x)
{
  return x;
}

//! Unary negative operator.
template<typename _T, std::size_t N>
inline
__device__
__host__
array<_T, N>
operator-(const array<_T, N>& x)
{
  array<_T, N> y(x);
  for (std::size_t n = 0; n != N; ++n) {
    y[n] = -y[n];
  }
  return y;
}

//@}
//-------------------------------------------------------------------------
/*! \defgroup cudaArrayBinary Binary Operators
  These functions define binary operators for arrays. The operands may be
  arrays or scalars.

  \verbatim
  cuda::array<double, 3> x, y;
  ...
  // Addition
  x = x + y;
  x = x + 2.;
  x = 2. + x;
  // Subtraction.
  x = x - y;
  x = x - 2.;
  x = 2. - x;
  // Multiplication.
  x = x * y;
  x = x * 2.;
  x = 2. * x;
  // Division.
  x = x / y;
  x = x / 2.;
  x = 2. / x; \endverbatim
  \verbatim
  cuda::array<unsigned, 3> x, y;
  ...
  // Modulus.
  x = x % y;
  x = x % 2;
  x = 2 % x; \endverbatim

  \note Because these function instantiate cuda::array objects,
  they are not as efficient
  as their corresponding \ref cudaArrayAssignmentArray "assignment operators."
  For example, the following
  \verbatim
  cuda::array<double, 3> center;
  double radius;
  ...
  center += radius; \endverbatim
  is more efficient than
  \verbatim
  center = center + radius; \endverbatim
*/
//@{

//! Array-scalar addition.
template<typename _T, typename _U, std::size_t N>
inline
__device__
__host__
auto
operator+(const array<_T, N>& x, const _U& y)
{
  array<typename std::remove_const<decltype(x[0] + y)>::type, N> z;
  for (std::size_t i = 0; i != N; ++i) {
    z[i] = x[i] + y;
  }
  return z;
}

//! Scalar-Array addition.
template<typename _T, typename _U, std::size_t N>
inline
__device__
__host__
auto
operator+(const _T& x, const array<_U, N>& y)
{
  array<typename std::remove_const<decltype(x + y[0])>::type, N> z;
  for (std::size_t i = 0; i != N; ++i) {
    z[i] = x + y[i];
  }
  return z;
}

//! Array-array addition.
template<typename _T, typename _U, std::size_t N>
inline
__device__
__host__
auto
operator+(const array<_T, N>& x, const array<_U, N>& y)
{
  array<typename std::remove_const<decltype(x[0] + y[0])>::type, N> z;
  for (std::size_t i = 0; i != N; ++i) {
    z[i] = x[i] + y[i];
  }
  return z;
}

//! Array-scalar subtraction.
template<typename _T, typename _U, std::size_t N>
inline
__device__
__host__
auto
operator-(const array<_T, N>& x, const _U& y)
{
  array<typename std::remove_const<decltype(x[0] + y)>::type, N> z;
  for (std::size_t i = 0; i != N; ++i) {
    z[i] = x[i] - y;
  }
  return z;
}

//! Scalar-Array subtraction.
template<typename _T, typename _U, std::size_t N>
inline
__device__
__host__
auto
operator-(const _T& x, const array<_U, N>& y)
{
  array<typename std::remove_const<decltype(x + y[0])>::type, N> z;
  for (std::size_t i = 0; i != N; ++i) {
    z[i] = x - y[i];
  }
  return z;
}

//! Array-array subtraction.
template<typename _T, typename _U, std::size_t N>
inline
__device__
__host__
auto
operator-(const array<_T, N>& x, const array<_U, N>& y)
{
  array<typename std::remove_const<decltype(x[0] + y[0])>::type, N> z;
  for (std::size_t i = 0; i != N; ++i) {
    z[i] = x[i] - y[i];
  }
  return z;
}

//! Array-scalar multiplication.
template<typename _T, typename _U, std::size_t N>
inline
__device__
__host__
auto
operator*(const array<_T, N>& x, const _U& y)
{
  array<typename std::remove_const<decltype(x[0] + y)>::type, N> z;
  for (std::size_t i = 0; i != N; ++i) {
    z[i] = x[i] * y;
  }
  return z;
}

//! Scalar-Array multiplication.
template<typename _T, typename _U, std::size_t N>
inline
__device__
__host__
auto
operator*(const _T& x, const array<_U, N>& y)
{
  array<typename std::remove_const<decltype(x + y[0])>::type, N> z;
  for (std::size_t i = 0; i != N; ++i) {
    z[i] = x * y[i];
  }
  return z;
}

//! Array-array multiplication.
template<typename _T, typename _U, std::size_t N>
inline
__device__
__host__
auto
operator*(const array<_T, N>& x, const array<_U, N>& y)
{
  array<typename std::remove_const<decltype(x[0] + y[0])>::type, N> z;
  for (std::size_t i = 0; i != N; ++i) {
    z[i] = x[i] * y[i];
  }
  return z;
}

//! Array-scalar division.
template<typename _T, typename _U, std::size_t N>
inline
__device__
__host__
auto
operator/(const array<_T, N>& x, const _U& y)
{
#if defined(STLIB_DEBUG) && ! defined(__CUDA_ARCH__)
  assert(y != 0);
#endif
  array<typename std::remove_const<decltype(x[0] + y)>::type, N> z;
  for (std::size_t i = 0; i != N; ++i) {
    z[i] = x[i] / y;
  }
  return z;
}

//! Scalar-Array division.
template<typename _T, typename _U, std::size_t N>
inline
__device__
__host__
auto
operator/(const _T& x, const array<_U, N>& y)
{
  array<typename std::remove_const<decltype(x + y[0])>::type, N> z;
  for (std::size_t i = 0; i != N; ++i) {
#if defined(STLIB_DEBUG) && ! defined(__CUDA_ARCH__)
    assert(y[i] != 0);
#endif
    z[i] = x / y[i];
  }
  return z;
}

//! Array-array division.
template<typename _T, typename _U, std::size_t N>
inline
__device__
__host__
auto
operator/(const array<_T, N>& x, const array<_U, N>& y)
{
  array<typename std::remove_const<decltype(x[0] + y[0])>::type, N> z;
  for (std::size_t i = 0; i != N; ++i) {
#if defined(STLIB_DEBUG) && ! defined(__CUDA_ARCH__)
    assert(y[i] != 0);
#endif
    z[i] = x[i] / y[i];
  }
  return z;
}

//! Array-scalar modulus.
template<typename _T, typename _U, std::size_t N>
inline
__device__
__host__
auto
operator%(const array<_T, N>& x, const _U& y)
{
#if defined(STLIB_DEBUG) && ! defined(__CUDA_ARCH__)
  assert(y != 0);
#endif
  array<typename std::remove_const<decltype(x[0] + y)>::type, N> z;
  for (std::size_t i = 0; i != N; ++i) {
    z[i] = x[i] % y;
  }
  return z;
}

//! Scalar-Array modulus.
template<typename _T, typename _U, std::size_t N>
inline
__device__
__host__
auto
operator%(const _T& x, const array<_U, N>& y)
{
  array<typename std::remove_const<decltype(x + y[0])>::type, N> z;
  for (std::size_t i = 0; i != N; ++i) {
#if defined(STLIB_DEBUG) && ! defined(__CUDA_ARCH__)
    assert(y[i] != 0);
#endif
    z[i] = x % y[i];
  }
  return z;
}

//! Array-array modulus.
template<typename _T, typename _U, std::size_t N>
inline
__device__
__host__
auto
operator%(const array<_T, N>& x, const array<_U, N>& y)
{
  array<typename std::remove_const<decltype(x[0] + y[0])>::type, N> z;
  for (std::size_t i = 0; i != N; ++i) {
#if defined(STLIB_DEBUG) && ! defined(__CUDA_ARCH__)
    assert(y[i] != 0);
#endif
    z[i] = x[i] % y[i];
  }
  return z;
}







#if 0
//-------------------------------------------------------------------------
// Old implementation using Loki typelist.

//! The result type for an arithmetic expression.
/*!
  \todo This is a dirty hack and does not work for arbitrary types.
  Use the typeof versions when it makes its way into the C++ standard.
*/
template<typename _X, typename _Y>
struct ArithmeticResult {
  //! An ordered list of numeric types.
  typedef LOKI_TYPELIST_10(double, float, unsigned long, long, unsigned, int,
                           unsigned short, short, unsigned char, signed char)
  OrderedTypes;
  //! The result type.
  typedef typename boost::mpl::if_c<int(Loki::TL::IndexOf<OrderedTypes, _X>::value)
  < int(Loki::TL::IndexOf<OrderedTypes, _Y>::value),
  _X, _Y>::type Type;
};


//! Array-scalar addition.
template<typename _T, typename _U, std::size_t N>
inline
__device__
__host__
array<typename ArithmeticResult<_T, _U>::Type, N>
operator+(const array<_T, N>& x, const _U& y)
{
  array<typename ArithmeticResult<_T, _U>::Type, N> z;
  for (std::size_t i = 0; i != N; ++i) {
    z[i] = x[i] + y;
  }
  return z;
}

//! Scalar-Array addition.
template<typename _T, typename _U, std::size_t N>
inline
__device__
__host__
array<typename ArithmeticResult<_T, _U>::Type, N>
operator+(const _T& x, const array<_U, N>& y)
{
  array<typename ArithmeticResult<_T, _U>::Type, N> z;
  for (std::size_t i = 0; i != N; ++i) {
    z[i] = x + y[i];
  }
  return z;
}

//! Array-array addition.
template<typename _T, typename _U, std::size_t N>
inline
__device__
__host__
array<typename ArithmeticResult<_T, _U>::Type, N>
operator+(const array<_T, N>& x, const array<_U, N>& y)
{
  array<typename ArithmeticResult<_T, _U>::Type, N> z;
  for (std::size_t i = 0; i != N; ++i) {
    z[i] = x[i] + y[i];
  }
  return z;
}

//! Array-scalar subtraction.
template<typename _T, typename _U, std::size_t N>
inline
__device__
__host__
array<typename ArithmeticResult<_T, _U>::Type, N>
operator-(const array<_T, N>& x, const _U& y)
{
  array<typename ArithmeticResult<_T, _U>::Type, N> z;
  for (std::size_t i = 0; i != N; ++i) {
    z[i] = x[i] - y;
  }
  return z;
}

//! Scalar-Array subtraction.
template<typename _T, typename _U, std::size_t N>
inline
__device__
__host__
array<typename ArithmeticResult<_T, _U>::Type, N>
operator-(const _T& x, const array<_U, N>& y)
{
  array<typename ArithmeticResult<_T, _U>::Type, N> z;
  for (std::size_t i = 0; i != N; ++i) {
    z[i] = x - y[i];
  }
  return z;
}

//! Array-array subtraction.
template<typename _T, typename _U, std::size_t N>
inline
__device__
__host__
array<typename ArithmeticResult<_T, _U>::Type, N>
operator-(const array<_T, N>& x, const array<_U, N>& y)
{
  array<typename ArithmeticResult<_T, _U>::Type, N> z;
  for (std::size_t i = 0; i != N; ++i) {
    z[i] = x[i] - y[i];
  }
  return z;
}

//! Array-scalar multiplication.
template<typename _T, typename _U, std::size_t N>
inline
__device__
__host__
array<typename ArithmeticResult<_T, _U>::Type, N>
operator*(const array<_T, N>& x, const _U& y)
{
  array<typename ArithmeticResult<_T, _U>::Type, N> z;
  for (std::size_t i = 0; i != N; ++i) {
    z[i] = x[i] * y;
  }
  return z;
}

//! Scalar-Array multiplication.
template<typename _T, typename _U, std::size_t N>
inline
__device__
__host__
array<typename ArithmeticResult<_T, _U>::Type, N>
operator*(const _T& x, const array<_U, N>& y)
{
  array<typename ArithmeticResult<_T, _U>::Type, N> z;
  for (std::size_t i = 0; i != N; ++i) {
    z[i] = x * y[i];
  }
  return z;
}

//! Array-array multiplication.
template<typename _T, typename _U, std::size_t N>
inline
__device__
__host__
array<typename ArithmeticResult<_T, _U>::Type, N>
operator*(const array<_T, N>& x, const array<_U, N>& y)
{
  array<typename ArithmeticResult<_T, _U>::Type, N> z;
  for (std::size_t i = 0; i != N; ++i) {
    z[i] = x[i] * y[i];
  }
  return z;
}

//! Array-scalar division.
template<typename _T, typename _U, std::size_t N>
inline
__device__
__host__
array<typename ArithmeticResult<_T, _U>::Type, N>
operator/(const array<_T, N>& x, const _U& y)
{
#if defined(STLIB_DEBUG) && ! defined(__CUDA_ARCH__)
  assert(y != 0);
#endif
  array<typename ArithmeticResult<_T, _U>::Type, N> z;
  for (std::size_t i = 0; i != N; ++i) {
    z[i] = x[i] / y;
  }
  return z;
}

//! Scalar-Array division.
template<typename _T, typename _U, std::size_t N>
inline
__device__
__host__
array<typename ArithmeticResult<_T, _U>::Type, N>
operator/(const _T& x, const array<_U, N>& y)
{
  array<typename ArithmeticResult<_T, _U>::Type, N> z;
  for (std::size_t i = 0; i != N; ++i) {
#if defined(STLIB_DEBUG) && ! defined(__CUDA_ARCH__)
    assert(y[i] != 0);
#endif
    z[i] = x / y[i];
  }
  return z;
}

//! Array-array division.
template<typename _T, typename _U, std::size_t N>
inline
__device__
__host__
array<typename ArithmeticResult<_T, _U>::Type, N>
operator/(const array<_T, N>& x, const array<_U, N>& y)
{
  array<typename ArithmeticResult<_T, _U>::Type, N> z;
  for (std::size_t i = 0; i != N; ++i) {
#if defined(STLIB_DEBUG) && ! defined(__CUDA_ARCH__)
    assert(y[i] != 0);
#endif
    z[i] = x[i] / y[i];
  }
  return z;
}

//! Array-scalar modulus.
template<typename _T, typename _U, std::size_t N>
inline
__device__
__host__
array<typename ArithmeticResult<_T, _U>::Type, N>
operator%(const array<_T, N>& x, const _U& y)
{
#if defined(STLIB_DEBUG) && ! defined(__CUDA_ARCH__)
  assert(y != 0);
#endif
  array<typename ArithmeticResult<_T, _U>::Type, N> z;
  for (std::size_t i = 0; i != N; ++i) {
    z[i] = x[i] % y;
  }
  return z;
}

//! Scalar-Array modulus.
template<typename _T, typename _U, std::size_t N>
inline
__device__
__host__
array<typename ArithmeticResult<_T, _U>::Type, N>
operator%(const _T& x, const array<_U, N>& y)
{
  array<typename ArithmeticResult<_T, _U>::Type, N> z;
  for (std::size_t i = 0; i != N; ++i) {
#if defined(STLIB_DEBUG) && ! defined(__CUDA_ARCH__)
    assert(y[i] != 0);
#endif
    z[i] = x % y[i];
  }
  return z;
}

//! Array-array modulus.
template<typename _T, typename _U, std::size_t N>
inline
__device__
__host__
array<typename ArithmeticResult<_T, _U>::Type, N>
operator%(const array<_T, N>& x, const array<_U, N>& y)
{
  array<typename ArithmeticResult<_T, _U>::Type, N> z;
  for (std::size_t i = 0; i != N; ++i) {
#if defined(STLIB_DEBUG) && ! defined(__CUDA_ARCH__)
    assert(y[i] != 0);
#endif
    z[i] = x[i] % y[i];
  }
  return z;
}

#endif

//@}
//-------------------------------------------------------------------------
/*! \defgroup cudaArrayFile File I/O

  These functions write and read cuda::array's in ascii and binary format.
  The file format is simply the sequence of elements, the number of elements
  is not read or written.
  \verbatim
  cuda::array<double, 3> x;
  ...
  // Ascii.
  std::cin >> x;
  std::cout << x;
  // Binary.
  std::ifstream in("input.bin");
  read(in, &x);
  std::ofstream out("output.bin");
  write(out, x); \endverbatim
*/
//@{

#ifndef __CUDA_ARCH__

//! Write the space-separated elements.
/*!
  Format:
  \verbatim
  x[0] x[1] x[2] ... \endverbatim
*/
template<typename _T, std::size_t N>
inline
__host__
std::ostream&
operator<<(std::ostream& out, const array<_T, N>& x)
{
  std::copy(x.begin(), x.end(), std::ostream_iterator<_T>(out, " "));
  return out;
}

//! Read the elements.
template<typename _T, std::size_t N>
inline
__host__
std::istream&
operator>>(std::istream& in, array<_T, N>& x)
{
  for (std::size_t n = 0; n != x.size(); ++n) {
    in >> x[n];
  }
  return in;
}

//! Write the elements in binary format.
template<typename _T, std::size_t N>
inline
__host__
void
write(const array<_T, N>& x, std::ostream& out)
{
  out.write(reinterpret_cast<const char*>(&x), sizeof(array<_T, N>));
}

//! Read the elements in binary format.
template<typename _T, std::size_t N>
inline
__host__
void
read(array<_T, N>* x, std::istream& in)
{
  in.read(reinterpret_cast<char*>(x), sizeof(array<_T, N>));
}

#endif

//@}
//-------------------------------------------------------------------------
/*! \defgroup cudaArrayMathematical Mathematical Functions
  These functions define some common mathematical operations on
  array's. There are utility functions for the sum, product,
  minimum, maximum, etc.
  \verbatim
  cuda::array<double, 3> x, y, z;
  ...
  // Sum.
  const double total = sum(x);
  // Product.
  const double volume = product(x);
  // Minimum.
  const double minValue = min(x);
  // Maximum.
  const double maxValue = max(x);
  // Element-wise minimum.
  z = min(x, y);
  // Element-wise maximum.
  z = min(x, y);
  // Existence of a value.
  const bool hasNull = hasElement(x, 0);
  // Existence and index of a value.
  std::size_t i;
  const bool hasOne = hasElement(x, 1, &i);
  // Index of a value.
  i = index(x); \endverbatim

  There are also function for treating a cuda::array as a Cartesian
  point or vector.
  \verbatim
  cuda::array<double, 3> x, y, z;
  ...
  // Dot product.
  const double d = dot(x, y);
  // Cross product.
  z = cross(x, y);
  // Cross product that avoids constructing an array.
  cross(x, y, &z);
  // Triple product.
  const double volume = tripleProduct(x, y, z);
  // Discriminant.
  cuda::array<double, 2> a, b;
  const double disc = discriminant(a, b);
  // Squared magnitude.
  const double sm = squaredMagnitude(x);
  // Magnitude.
  const double m = magnitude(x);
  // Normalize a vector.
  normalize(&x);
  // Negate a vector.
  negateElements(&x);
  // Squared distance.
  const double d2 = squaredDistance(x, y);
  // Distance.
  const double d = euclideanDistance(x, y); \endverbatim
*/
//@{

//! Return the sum of the components.
template<typename _T, std::size_t N>
inline
__device__
__host__
_T
sum(const array<_T, N>& x)
{
  _T result = 0;
  for (std::size_t i = 0; i != N; ++i) {
    result += x[i];
  }
  return result;
}

//! Return the product of the components.
template<typename _T, std::size_t N>
inline
__device__
__host__
_T
product(const array<_T, N>& x)
{
  _T result = 1;
  for (std::size_t i = 0; i != N; ++i) {
    result *= x[i];
  }
  return result;
}

//! Return the minimum component.  Use < for comparison.
template<typename _T, std::size_t N>
inline
__device__
__host__
_T
min(const array<_T, N>& x)
{
#if defined(STLIB_DEBUG) && ! defined(__CUDA_ARCH__)
  assert(x.size() != 0);
#endif
  _T result = x[0];
  for (std::size_t i = 1; i != N; ++i) {
    if (x[i] < result) {
      result = x[i];
    }
  }
  return result;
}

//! Return the maximum component.  Use > for comparison.
template<typename _T, std::size_t N>
inline
__device__
__host__
_T
max(const array<_T, N>& x)
{
#if defined(STLIB_DEBUG) && ! defined(__CUDA_ARCH__)
  assert(x.size() != 0);
#endif
  _T result = x[0];
  for (std::size_t i = 1; i != N; ++i) {
    if (x[i] > result) {
      result = x[i];
    }
  }
  return result;
}

//! Return an array that is element-wise the minimum of the two.
template<typename _T, std::size_t N>
inline
__device__
__host__
array<_T, N>
min(const array<_T, N>& x, const array<_T, N>& y)
{
  array<_T, N> z;
  for (std::size_t n = 0; n != N; ++n) {
    z[n] = ::min(x[n], y[n]);
  }
  return z;
}

//! Return an array that is element-wise the maximum of the two.
template<typename _T, std::size_t N>
inline
__device__
__host__
array<_T, N>
max(const array<_T, N>& x, const array<_T, N>& y)
{
  array<_T, N> z;
  for (std::size_t n = 0; n != N; ++n) {
    z[n] = ::max(x[n], y[n]);
  }
  return z;
}

//! Return true if the array has the specified element.
template<typename _T, std::size_t N, typename _Comparable>
inline
__device__
__host__
bool
hasElement(const array<_T, N>& x, const _Comparable& a)
{
  for (std::size_t i = 0; i != N; ++i) {
    if (x[i] == a) {
      return true;
    }
  }
  return false;
}

//! Return true if the array has the specified element.
/*!
  If true, compute the index of the elementt.
*/
template<typename _T, std::size_t N, typename _Comparable>
inline
__device__
__host__
bool
hasElement(const array<_T, N>& x, const _Comparable& a, std::size_t* i)
{
  for (*i = 0; *i != x.size(); ++*i) {
    if (a == x[*i]) {
      return true;
    }
  }
  return false;
}

//! Return the index of the specified element. Return std::numeric_limits<std::size_t>::max() if the element is not in the array.
template<typename _T, std::size_t N, typename _Comparable>
inline
__device__
__host__
std::size_t
index(const array<_T, N>& x, const _Comparable& a)
{
  for (std::size_t i = 0; i != x.size(); ++i) {
    if (a == x[i]) {
      return i;
    }
  }
  return std::numeric_limits<std::size_t>::max();
}

//! Return the dot product of the two arrays.
template<typename _T, std::size_t N>
inline
__device__
__host__
_T
dot(const array<_T, N>& x, const array<_T, N>& y)
{
  _T p = 0;
  for (std::size_t i = 0; i != N; ++i) {
    p += x[i] * y[i];
  }
  return p;
}

//! Return the dot product of the two arrays.
/*! This specialization is a little faster than the dimension-general code. */
template<typename _T>
inline
__device__
__host__
_T
dot(const array<_T, 3>& x, const array<_T, 3>& y)
{
  return x[0] * y[0] + x[1] * y[1] + x[2] * y[2];
}

//! Return the cross product of the two arrays.
template<typename _T>
inline
__device__
__host__
array<_T, 3>
cross(const array<_T, 3>& x, const array<_T, 3>& y)
{
  array<_T, 3> result = {{
      x[1]* y[2] - y[1]* x[2],
      y[0]* x[2] - x[0]* y[2],
      x[0]* y[1] - y[0]* x[1]
    }
  };
  return result;
}

//! Return the cross product and derivative of cross product of the two arrays.
template<typename _T>
inline
__device__
__host__
array<_T, 3>
cross(const array<_T, 3>& x, const array<_T, 3>& y,
      array<array<_T, 3>, 3>* dx, array<array<_T, 3>, 3>* dy)
{
  (*dx)[0][0] = 0.0;
  (*dx)[0][1] = y[2];
  (*dx)[0][2] = -y[1];
  (*dx)[1][0] = -y[2];
  (*dx)[1][1] = 0.0;
  (*dx)[1][2] = y[0];
  (*dx)[2][0] = y[1];
  (*dx)[2][1] = -y[0];
  (*dx)[2][2] = 0.0;
  (*dy)[0][0] = 0.0;
  (*dy)[0][1] = -x[2];
  (*dy)[0][2] = x[1];
  (*dy)[1][0] = x[2];
  (*dy)[1][1] = 0.0;
  (*dy)[1][2] = -x[0];
  (*dy)[2][0] = -x[1];
  (*dy)[2][1] = x[0];
  (*dy)[2][2] = 0.0;
  //*dx = {{ {{ 0.0,   y[2],  -y[1] }},
  //         {{ -y[2], 0.0,   y[0]  }},
  //         {{ y[1],  -y[0], 0.0   }} }};
  //*dy = {{ {{ 0.0,   -x[2], x[1]  }},
  //         {{ x[2],  0.0,   -x[0] }},
  //         {{ -x[1], x[0],  0.0   }} }};
  array<_T, 3> result = {{
      x[1]* y[2] - y[1]* x[2],
      y[0]* x[2] - x[0]* y[2],
      x[0]* y[1] - y[0]* x[1]
    }
  };
  return result;
}

//! Compute the cross product of the two arrays.
template<typename _T>
inline
__device__
__host__
void
cross(const array<_T, 3>& x, const array<_T, 3>& y, array<_T, 3>* result)
{
  (*result)[0] = x[1] * y[2] - y[1] * x[2];
  (*result)[1] = y[0] * x[2] - x[0] * y[2];
  (*result)[2] = x[0] * y[1] - y[0] * x[1];
}

//! Return the triple product of the three arrays.
template<typename _T>
inline
__device__
__host__
_T
tripleProduct(const array<_T, 3>& x, const array<_T, 3>& y,
              const array<_T, 3>& z)
{
  array<_T, 3> t;
  cross(y, z, &t);
  return dot(x, t);
}

//! Return the discriminant of the two arrays.
template<typename _T>
inline
__device__
__host__
_T
discriminant(const array<_T, 2>& x, const array<_T, 2>& y)
{
  return x[0] * y[1] - x[1] * y[0];
}

//! Return the squared magnitude.
template<typename _T, std::size_t N>
inline
__device__
__host__
_T
squaredMagnitude(const array<_T, N>& x)
{
  return dot(x, x);
}

//! Return the magnitude.
template<std::size_t N>
inline
__device__
__host__
float
magnitude(const array<float, N>& x)
{
  return sqrtf(squaredMagnitude(x));
}

//! Return the magnitude.
template<std::size_t N>
inline
__device__
__host__
double
magnitude(const array<double, N>& x)
{
  return sqrt(squaredMagnitude(x));
}

//! Normalize the vector to have unit magnitude.
template<typename _T, std::size_t N>
inline
__device__
__host__
void
normalize(array<_T, N>* x)
{
  const _T mag = magnitude(*x);
  if (mag != 0) {
    *x /= mag;
  }
  else {
    // If the vector has zero length, choose the unit vector whose first
    // coordinate is 1.
    (*x)[0] = 1;
    for (std::size_t i = 0; i != N; ++i) {
      (*x)[i] = 0;
    }
  }
}

//! Negate the vector.
template<typename _T, std::size_t N>
inline
__device__
__host__
void
negateElements(array<_T, N>* x)
{
  for (std::size_t n = 0; n != N; ++n) {
    (*x)[n] = - (*x)[n];
  }
}

//! Negate the vector.
template<size_t N>
inline
__device__
__host__
void
negateElements(array<bool, N>* x)
{
  for (std::size_t n = 0; n != N; ++n) {
    (*x)[n] = !(*x)[n];
  }
}

//! Return the squared distance between the two points.
template<typename _T, std::size_t N>
inline
__device__
__host__
_T
squaredDistance(const array<_T, N>& x, const array<_T, N>& y)
{
  _T d = 0;
  for (std::size_t n = 0; n != N; ++n) {
    d += (x[n] - y[n]) * (x[n] - y[n]);
  }
  return d;
}

//! Return the squared distance between the two points.
/*! This specialization is a little faster than the dimension-general code. */
template<typename _T>
inline
__device__
__host__
_T
squaredDistance(const array<_T, 3>& x, const array<_T, 3>& y)
{
  return (x[0] - y[0]) * (x[0] - y[0]) +
         (x[1] - y[1]) * (x[1] - y[1]) +
         (x[2] - y[2]) * (x[2] - y[2]);
}

//! Return the Euclidean distance between the two points.
/*!
  \note distance would not be a good name for this function because
  std::distance() calculates the distance between iterators.
*/
template<std::size_t N>
inline
__device__
__host__
float
euclideanDistance(const array<float, N>& x, const array<float, N>& y)
{
  return sqrtf(squaredDistance(x, y));
}

//! Return the Euclidean distance between the two points.
/*!
  \note distance would not be a good name for this function because
  std::distance() calculates the distance between iterators.
*/
template<std::size_t N>
inline
__device__
__host__
double
euclideanDistance(const array<double, N>& x, const array<double, N>& y)
{
  return sqrt(squaredDistance(x, y));
}

//@}

} // namespace tr1
} // namespace std

#endif
