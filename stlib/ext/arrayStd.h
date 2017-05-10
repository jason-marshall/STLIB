// -*- C++ -*-

/**
  \file
  \brief Functions and operators for std::array.
*/

#if !defined(__ext_arrayStd_h__)
#define __ext_arrayStd_h__

#include <array>

#include <algorithm>
#include <numeric>
#include <iostream>
#include <iterator>
#include <limits>
#include <type_traits>

#include <cmath>
#include <cassert>

// SSE 4.1
#ifdef __SSE4_1__
#include <smmintrin.h>
#endif

/// Add using directives for the math operators for std::array.
#define USING_STLIB_EXT_ARRAY_MATH_OPERATORS    \
  using stlib::ext::operator+=;                 \
  using stlib::ext::operator-=;                 \
  using stlib::ext::operator*=;                 \
  using stlib::ext::operator/=;                 \
  using stlib::ext::operator%=;                 \
  using stlib::ext::operator<<=;                \
  using stlib::ext::operator>>=;                \
  using stlib::ext::operator+;                  \
  using stlib::ext::operator-;                  \
  using stlib::ext::operator*;                  \
  using stlib::ext::operator/;                  \
  using stlib::ext::operator%

/// Add using directives for the input and output operators for std::array.
#define USING_STLIB_EXT_ARRAY_IO_OPERATORS      \
  using stlib::ext::operator<<;                 \
  using stlib::ext::operator>>

/// Add using directives for the operators for std::array.
#define USING_STLIB_EXT_ARRAY                             \
  USING_STLIB_EXT_ARRAY_MATH_OPERATORS;                   \
  USING_STLIB_EXT_ARRAY_IO_OPERATORS

/**
\page extArray Extensions to std::array

Here we provide functions to extend the functionality of the std::array
class [\ref extBecker2007 "Becker, 2007"]. The functions are grouped into
the following categories:
- \ref extArrayAssignmentScalar
- \ref extArrayAssignmentArray
- \ref extArrayUnary
- \ref extArrayBinary
- \ref extArrayFile
- \ref extArrayMathematical

To use the operators, you will need to add using directives to your code.
You can either add them individually, or use one of the following convenience
macros.
\code
USING_STLIB_EXT_ARRAY_MATH_OPERATORS;
USING_STLIB_EXT_ARRAY_IO_OPERATORS;
USING_STLIB_EXT_ARRAY;
\endcode
*/

namespace stlib
{
namespace ext
{

//-------------------------------------------------------------------------
/** \defgroup extArrayAssignmentScalar Assignment Operators with a Scalar Operand.

  These functions apply an assignment operation to each element of the array.
  Thus you can add, subtract, etc. a scalar from each element. Note that 
  for +=, -=, *=, /=, and %=, the type of the scalar must match the value type.
  \code. For the shift-asignment operators, the rhs must be an \c int.
  USING_STLIB_EXT_ARRAY_MATH_OPERATORS;
  std::array<double, 3> x = {{0, 0, 0}};
  x += double(2);
  x -= 3.;
  x *= 5.;
  x /= 7.;
  \endcode

  \code
  std::array<unsigned, 3> x = {{0, 0, 0}};
  x %= unsigned(2);
  x <<= 3;
  x >>= 5;
  \endcode
*/
//@{

/// Array-scalar addition.
/**
   Note that one might want to generalize this to accept a rhs with a different
   number type. However, doing so would make the function too greedy. Making
   the rhs a separate templated type would not only match different number 
   types, it would match different object types. For example, T might be 
   a std::array<float, 3> and the rhs might be a double. Thus, generalizing
   this function would lead to confusing code.
*/
template<typename T, std::size_t N>
std::array<T, N>&
operator+=(std::array<T, N>& x, T const& value);

/// Array-scalar subtraction.
template<typename T, std::size_t N>
std::array<T, N>&
operator-=(std::array<T, N>& x, T const& value);

/// Array-scalar multiplication.
template<typename T, std::size_t N>
std::array<T, N>&
operator*=(std::array<T, N>& x, T const& value);

/// Array-scalar division.
template<typename T, std::size_t N>
std::array<T, N>&
operator/=(std::array<T, N>& x, T const& value);

/// Array-scalar modulus.
template<typename T, std::size_t N>
std::array<T, N>&
operator%=(std::array<T, N>& x, T const& value);

/// Left shift.
template<typename T, std::size_t N>
std::array<T, N>&
operator<<=(std::array<T, N>& x, int const offset);

/// Right shift.
template<typename T, std::size_t N>
std::array<T, N>&
operator>>=(std::array<T, N>& x, int const offset);

//@}
//-------------------------------------------------------------------------
/** \defgroup extArrayAssignmentArray Assignment Operators with an Array Operand.

  These functions define assignment operations between arrays. The assignment
  operation is applied element-wise.
  \code
  USING_STLIB_EXT_ARRAY_MATH_OPERATORS;
  std::array<double, 3> x, y;
  ...
  x += y;
  x -= y;
  x *= y;
  x /= y;
  \endcode
  \code
  std::array<unsigned, 3> x, y;
  ...
  x %= y;
  x <<= y;
  x >>= y;
  \endcode
*/
//@{

/// Array-array addition.
/**
   Note that the value types for the arrays must match. If we generalized this
   function to allow different number types, we would run into trouble.
   For example, the code below would produce unexpected results. Instead of
   adding the point to each vertex of the simplex, a different scalar 
   would be added to each point. That is, \c point[0] would be added to 
   each coordinate of the first vertex, \c point[1] would be added to 
   each coordinate of the second vertex, and so on.
   \code
   std::array<std::array<double, 3>, 3> simplex;
   std::array<double, 3> point;
   ...
   simplex += point;
   \endcode
*/
template<typename T, std::size_t N>
inline
std::array<T, N>&
operator+=(std::array<T, N>& x, std::array<T, N> const& y)
{
  for (std::size_t n = 0; n != x.size(); ++n) {
    x[n] += y[n];
  }
  return x;
}

/// Array-array subtraction.
template<typename T, std::size_t N>
inline
std::array<T, N>&
operator-=(std::array<T, N>& x, std::array<T, N> const& y)
{
  for (std::size_t n = 0; n != x.size(); ++n) {
    x[n] -= y[n];
  }
  return x;
}

/// Array-array multiplication.
template<typename T, std::size_t N>
inline
std::array<T, N>&
operator*=(std::array<T, N>& x, std::array<T, N> const& y)
{
  for (std::size_t n = 0; n != x.size(); ++n) {
    x[n] *= y[n];
  }
  return x;
}

/// Array-array division.
template<typename T, std::size_t N>
inline
std::array<T, N>&
operator/=(std::array<T, N>& x, std::array<T, N> const& y)
{
  for (std::size_t n = 0; n != x.size(); ++n) {
#ifdef STLIB_DEBUG
    assert(y[n] != 0);
#endif
    x[n] /= y[n];
  }
  return x;
}

/// Array-array modulus.
template<typename T, std::size_t N>
inline
std::array<T, N>&
operator%=(std::array<T, N>& x, std::array<T, N> const& y)
{
  for (std::size_t n = 0; n != x.size(); ++n) {
#ifdef STLIB_DEBUG
    assert(y[n] != 0);
#endif
    x[n] %= y[n];
  }
  return x;
}

/// Array-array left shift.
template<typename T, std::size_t N>
inline
std::array<T, N>&
operator<<=(std::array<T, N>& x, std::array<int, N> const& y)
{
  for (std::size_t n = 0; n != x.size(); ++n) {
    x[n] <<= y[n];
  }
  return x;
}

/// Array-array right shift.
template<typename T, std::size_t N>
inline
std::array<T, N>&
operator>>=(std::array<T, N>& x, std::array<int, N> const& y)
{
  for (std::size_t n = 0; n != x.size(); ++n) {
    x[n] >>= y[n];
  }
  return x;
}

//@}
//-------------------------------------------------------------------------
/** \defgroup extArrayUnary Unary Operators

  These functions define unary operations for arrays.
  \code
  USING_STLIB_EXT_ARRAY_MATH_OPERATORS;
  std::array<double, 3> x, y;
  ...
  x = +y;
  x = -y;
  \endcode
*/
//@{

/// Unary positive operator.
template<typename T, std::size_t N>
inline
std::array<T, N> const&
operator+(std::array<T, N> const& x)
{
  return x;
}

/// Unary negative operator.
template<typename T, std::size_t N>
inline
std::array<T, N>
operator-(std::array<T, N> const& x)
{
  std::array<T, N> y;
  for (std::size_t n = 0; n != N; ++n) {
    y[n] = -x[n];
  }
  return y;
}

//@}
//-------------------------------------------------------------------------
/** \defgroup extArrayBinary Binary Operators
  These functions define binary operators for arrays. The operands may be
  arrays or scalars.

  \code
  USING_STLIB_EXT_ARRAY_MATH_OPERATORS;
  std::array<double, 3> x, y;
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
  x = 2. / x;
  \endcode
  \code
  std::array<unsigned, 3> x, y;
  ...
  // Modulus.
  x = x % y;
  x = x % 2;
  x = 2 % x;
  \endcode

  \note Because these function instantiate std::array objects,
  they are not as efficient
  as their corresponding \ref extArrayAssignmentArray "assignment operators."
  For example, the following
  \code
  std::array<double, 3> center;
  double radius;
  ...
  center += radius;
  \endcode
  is more efficient than
  \code
  center = center + radius;
  \endcode
*/
//@{


/// Array-scalar addition.
template<typename T, std::size_t N>
inline
std::array<T, N>
operator+(std::array<T, N> x, T const& y)
{
  return x += y;
}

/// Scalar-Array addition.
template<typename T, std::size_t N>
inline
std::array<T, N>
operator+(T const& x, std::array<T, N> y)
{
  return y += x;
}

/// Array-array addition.
template<typename T, std::size_t N>
inline
std::array<T, N>
operator+(std::array<T, N> x, std::array<T, N> const& y)
{
  return x += y;
}

/// Array-scalar subtraction.
template<typename T, std::size_t N>
inline
std::array<T, N>
operator-(std::array<T, N> x, T const& y)
{
  return x -= y;
}

/// Scalar-Array subtraction.
template<typename T, std::size_t N>
inline
std::array<T, N>
operator-(T const& x, std::array<T, N> const& y)
{
  std::array<T, N> result;
  result.fill(x);
  return result -= y;
}

/// Array-array subtraction.
template<typename T, std::size_t N>
inline
std::array<T, N>
operator-(std::array<T, N> x, std::array<T, N> const& y)
{
  return x -= y;
}

/// Array-scalar multiplication.
template<typename T, std::size_t N>
inline
std::array<T, N>
operator*(std::array<T, N> x, T const& y)
{
  return x *= y;
}

/// Scalar-Array multiplication.
template<typename T, std::size_t N>
inline
std::array<T, N>
operator*(T const& x, std::array<T, N> y)
{
  return y *= x;
}

/// Array-array multiplication.
template<typename T, std::size_t N>
inline
std::array<T, N>
operator*(std::array<T, N> x, std::array<T, N> const& y)
{
  return x *= y;
}

/// Array-scalar division.
template<typename T, std::size_t N>
inline
std::array<T, N>
operator/(std::array<T, N> x, T const& y)
{
  return x /= y;
}

/// Scalar-Array division.
template<typename T, std::size_t N>
inline
std::array<T, N>
operator/(T const& x, std::array<T, N> const& y)
{
  std::array<T, N> result;
  result.fill(x);
  return result /= y;
}

/// Array-array division.
template<typename T, std::size_t N>
inline
std::array<T, N>
operator/(std::array<T, N> x, std::array<T, N> const& y)
{
  return x /= y;
}

/// Array-scalar modulus.
template<typename T, std::size_t N>
inline
std::array<T, N>
operator%(std::array<T, N> x, T const& y)
{
#ifdef STLIB_DEBUG
  assert(y != 0);
#endif
  return x %= y;
}

/// Scalar-Array modulus.
template<typename T, std::size_t N>
inline
std::array<T, N>
operator%(T const& x, std::array<T, N> const& y)
{
#ifdef STLIB_DEBUG
  for (std::size_t i = 0; i != N; ++i) {
    assert(y[i] != 0);
  }
#endif
  std::array<T, N> result;
  result.fill(x);
  return result %= y;
}

/// Array-array modulus.
template<typename T, std::size_t N>
inline
std::array<T, N>
operator%(std::array<T, N> x, std::array<T, N> const& y)
{
#ifdef STLIB_DEBUG
  for (std::size_t i = 0; i != N; ++i) {
    assert(y[i] != 0);
  }
#endif
  return x %= y;
}

//@}
//-------------------------------------------------------------------------
/** \defgroup extArrayFile File I/O

  These functions write and read std::array's in ascii and binary format.
  The file format is simply the sequence of elements, the number of elements
  is not read or written.
  \code
  USING_STLIB_EXT_ARRAY_IO_OPERATORS;
  std::array<double, 3> x;
  ...
  // Ascii.
  std::cin >> x;
  std::cout << x;
  // Binary.
  std::ifstream in("input.bin");
  stlib::ext::read(in, &x);
  std::ofstream out("output.bin");
  stlib::ext::write(out, x);
  \endcode
*/
//@{

/// Write the space-separated elements.
/**
  Format:
  \code
  x[0] x[1] x[2] ...
  \endcode
*/
template<typename T, std::size_t N>
inline
std::ostream&
operator<<(std::ostream& out, std::array<T, N> const& x)
{
  for (auto const& element: x) {
    out << element << ' ';
  }
  return out;
}

/// Read the elements.
template<typename T, std::size_t N>
inline
std::istream&
operator>>(std::istream& in, std::array<T, N>& x)
{
  for (std::size_t n = 0; n != x.size(); ++n) {
    in >> x[n];
  }
  return in;
}

/// Write the elements in binary format.
template<typename T, std::size_t N>
inline
void
write(std::array<T, N> const& x, std::ostream& out)
{
  out.write(reinterpret_cast<const char*>(&x), sizeof(std::array<T, N>));
}

/// Read the elements in binary format.
template<typename T, std::size_t N>
inline
void
read(std::array<T, N>* x, std::istream& in)
{
  in.read(reinterpret_cast<char*>(x), sizeof(std::array<T, N>));
}

//@}
//-------------------------------------------------------------------------
/** \defgroup extArrayMathematical Mathematical Functions
  These functions define some common mathematical operations on
  std::array's. There are utility functions for the sum, product,
  minimum, maximum, etc.
  \code
  USING_STLIB_EXT_ARRAY_MATH_OPERATORS;
  std::array<double, 3> x, y, z;
  ...
  // Sum.
  double const total = stlib::ext::sum(x);
  // Product.
  double const volume = stlib::ext::product(x);
  // Minimum.
  double const minValue = stlib::ext::min(x);
  // Maximum.
  double const maxValue = stlib::ext::max(x);
  // Element-wise minimum.
  z = stlib::ext::min(x, y);
  // Element-wise maximum.
  z = stlib::ext::min(x, y);
  // Existence of a value.
  bool const hasNull = stlib::ext::hasElement(x, 0);
  // Existence and index of a value.
  std::size_t i;
  bool const hasOne = stlib::ext::hasElement(x, 1, &i);
  // Index of a value.
  i = stlib::ext::index(x);
  \endcode

  There are also functions for treating a std::array as a Cartesian
  point or vector.
  \code
  std::array<double, 3> x, y, z;
  ...
  // Dot product.
  double const d = stlib::ext::dot(x, y);
  // Cross product.
  z = stlib::ext::cross(x, y);
  // Cross product that avoids constructing an array.
  stlib::ext::cross(x, y, &z);
  // Triple product.
  double const volume = stlib::ext::tripleProduct(x, y, z);
  // Discriminant.
  std::array<double, 2> a, b;
  double const disc = stlib::ext::discriminant(a, b);
  // Squared magnitude.
  double const sm = stlib::ext::squaredMagnitude(x);
  // Magnitude.
  double const m = stlib::ext::magnitude(x);
  // Normalize a vector.
  stlib::ext::normalize(&x);
  // Negate a vector.
  stlib::ext::negateElements(&x);
  // Squared distance.
  double const d2 = stlib::ext::squaredDistance(x, y);
  // Distance.
  double const d = stlib::ext::euclideanDistance(x, y);
  \endcode
*/
//@{

/// Return the sum of the components.
template<typename T, std::size_t N>
inline
T
sum(std::array<T, N> const& x)
{
  return std::accumulate(x.begin(), x.end(), T(0));
}

/// Return the product of the components.
template<typename T, std::size_t N>
inline
T
product(std::array<T, N> const& x)
{
  return std::accumulate(x.begin(), x.end(), T(1), std::multiplies<T>());
}

/// Return the minimum component.  Use < for comparison.
template<typename T, std::size_t N>
inline
T
min(std::array<T, N> const& x)
{
#ifdef STLIB_DEBUG
  assert(x.size() != 0);
#endif
  return *std::min_element(x.begin(), x.end());
}

/// Return the maximum component.  Use > for comparison.
template<typename T, std::size_t N>
inline
T
max(std::array<T, N> const& x)
{
#ifdef STLIB_DEBUG
  assert(x.size() != 0);
#endif
  return *std::max_element(x.begin(), x.end());
}

/// Return an array that is element-wise the minimum of the two.
template<typename T, std::size_t N>
inline
std::array<T, N>
min(std::array<T, N> const& x, std::array<T, N> const& y)
{
  std::array<T, N> z;
  for (std::size_t n = 0; n != N; ++n) {
    z[n] = std::min(x[n], y[n]);
  }
  return z;
}

/// Return an array that is element-wise the maximum of the two.
template<typename T, std::size_t N>
inline
std::array<T, N>
max(std::array<T, N> const& x, std::array<T, N> const& y)
{
  std::array<T, N> z;
  for (std::size_t n = 0; n != N; ++n) {
    z[n] = std::max(x[n], y[n]);
  }
  return z;
}

/// Return true if the array has the specified element.
template<typename T, std::size_t N, typename _Comparable>
inline
bool
hasElement(std::array<T, N> const& x, const _Comparable& a)
{
  return std::count(x.begin(), x.end(), a);
}

/// Return true if the array has the specified element.
/**
  If true, compute the index of the elementt.
*/
template<typename T, std::size_t N, typename _Comparable>
inline
bool
hasElement(std::array<T, N> const& x, const _Comparable& a, std::size_t* i)
{
  for (*i = 0; *i != x.size(); ++*i) {
    if (a == x[*i]) {
      return true;
    }
  }
  return false;
}

/// Return the index of the specified element. Return std::numeric_limits<std::size_t>::max() if the element is not in the array.
template<typename T, std::size_t N, typename _Comparable>
inline
std::size_t
index(std::array<T, N> const& x, const _Comparable& a)
{
  for (std::size_t i = 0; i != x.size(); ++i) {
    if (T(a) == x[i]) {
      return i;
    }
  }
  return std::numeric_limits<std::size_t>::max();
}

/// Return the dot product of the two arrays.
template<typename T, std::size_t N>
inline
T
dot(std::array<T, N> const& x, std::array<T, N> const& y)
{
  // Clean version:
  // return std::inner_product(x.begin(), x.end(), y.begin(), T(0));
  // More efficient because of loop unrolling:
  T p = 0;
  for (std::size_t i = 0; i != N; ++i) {
    p += x[i] * y[i];
  }
  return p;
}

/// Return the dot product of the two arrays.
/** This specialization is a little faster than the dimension-general code. */
template<typename T>
inline
T
dot(std::array<T, 3> const& x, std::array<T, 3> const& y)
{
  return x[0] * y[0] + x[1] * y[1] + x[2] * y[2];
}

//
// Versions that use SIMD intrinsics.
//
#ifndef STLIB_NO_SIMD_INTRINSICS


#ifdef __SSE4_1__
/// Return the dot product of the two arrays.
/** Specialization for single-precision, 3-D. */
inline
float
dot(std::array<float, 3> const& x, std::array<float, 3> const& y)
{
  // CONTINUE: Implementent specialization for 16-byte aligned structure.
#if 0
  // Note: Using load is OK because I don't access the fourth element,
  // which might be NaN.
  __m128 d = _mm_dp_ps(_mm_load_ps(&x[0]), _mm_load_ps(&y[0]), 0x71);
#endif
  __m128 d = _mm_dp_ps(_mm_set_ps(0, x[2], x[1], x[0]),
                       _mm_set_ps(0, y[2], y[1], y[0]), 0x71);
  return *reinterpret_cast<const float*>(&d);
}
#endif


#ifdef __SSE4_1__
/// Return the dot product of the two arrays.
/** Specialization for single-precision, 4-D. */
inline
float
dot(std::array<float, 4> const& x, std::array<float, 4> const& y)
{
  // CONTINUE: Implementent specialization for 16-byte aligned structure.
#if 0
  __m128 d = _mm_dp_ps(_mm_load_ps(&x[0]), _mm_load_ps(&y[0]), 0xF1);
#endif
  __m128 d = _mm_dp_ps(_mm_loadu_ps(&x[0]), _mm_loadu_ps(&y[0]), 0xF1);
  return *reinterpret_cast<const float*>(&d);
}
#endif


#ifdef __SSE4_1__
/// Return the dot product of the two arrays.
/** Specialization for double-precision, 4-D. */
inline
double
dot(std::array<double, 4> const& x, std::array<double, 4> const& y)
{
  // CONTINUE: Implementent specialization for 16-byte aligned structure.
#if 0
  __m128d a = _mm_add_sd(_mm_dp_pd(_mm_load_pd(&x[0]), _mm_load_pd(&y[0]),
                                   0x31),
                         _mm_dp_pd(_mm_load_pd(&x[2]), _mm_load_pd(&y[2]),
                                   0x31));
#endif
  __m128d a = _mm_add_sd(_mm_dp_pd(_mm_loadu_pd(&x[0]), _mm_loadu_pd(&y[0]),
                                   0x31),
                         _mm_dp_pd(_mm_loadu_pd(&x[2]), _mm_loadu_pd(&y[2]),
                                   0x31));
  return *reinterpret_cast<const double*>(&a);
}
#endif


#endif //#ifndef STLIB_NO_SIMD_INTRINSICS


/// Return the cross product of the two arrays.
template<typename T>
inline
std::array<T, 3>
cross(std::array<T, 3> const& x, std::array<T, 3> const& y)
{
  std::array<T, 3> result = {{
      x[1]* y[2] - y[1]* x[2],
      y[0]* x[2] - x[0]* y[2],
      x[0]* y[1] - y[0]* x[1]
    }
  };
  return result;
}

/// Return the cross product and derivative of cross product of the two arrays.
template<typename T>
inline
std::array<T, 3>
cross(std::array<T, 3> const& x, std::array<T, 3> const& y,
      std::array<std::array<T, 3>, 3>* dx, std::array<std::array<T, 3>, 3>* dy)
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
  std::array<T, 3> result = {{
      x[1]* y[2] - y[1]* x[2],
      y[0]* x[2] - x[0]* y[2],
      x[0]* y[1] - y[0]* x[1]
    }
  };
  return result;
}

/// Compute the cross product of the two arrays.
template<typename T>
inline
void
cross(std::array<T, 3> const& x, std::array<T, 3> const& y, std::array<T, 3>* result)
{
  (*result)[0] = x[1] * y[2] - y[1] * x[2];
  (*result)[1] = y[0] * x[2] - x[0] * y[2];
  (*result)[2] = x[0] * y[1] - y[0] * x[1];
}

/// Return the triple product of the three arrays.
template<typename T>
inline
T
tripleProduct(std::array<T, 3> const& x, std::array<T, 3> const& y,
              std::array<T, 3> const& z)
{
  std::array<T, 3> t;
  cross(y, z, &t);
  return dot(x, t);
}

/// Return the discriminant of the two arrays.
template<typename T>
inline
T
discriminant(std::array<T, 2> const& x, std::array<T, 2> const& y)
{
  return x[0] * y[1] - x[1] * y[0];
}

/// Return the squared magnitude.
template<typename T, std::size_t N>
inline
T
squaredMagnitude(std::array<T, N> const& x)
{
  return dot(x, x);
}

/// Return the magnitude.
template<typename T, std::size_t N>
inline
T
magnitude(std::array<T, N> const& x)
{
  return sqrt(squaredMagnitude(x));
}

/// Normalize the vector to have unit magnitude.
template<typename T, std::size_t N>
inline
void
normalize(std::array<T, N>* x)
{
  const T mag = magnitude(*x);
  if (mag != 0) {
    *x /= mag;
  }
  else {
    // If the vector has zero length, choose the unit vector whose first
    // coordinate is 1.
    std::fill(x->begin(), x->end(), T(0));
    (*x)[0] = 1;
  }
}

/// Negate the vector.
template<typename T, std::size_t N>
inline
void
negateElements(std::array<T, N>* x)
{
  for (std::size_t n = 0; n != N; ++n) {
    (*x)[n] = - (*x)[n];
  }
}

/// Negate the vector.
template<size_t N>
inline
void
negateElements(std::array<bool, N>* x)
{
  for (std::size_t n = 0; n != N; ++n) {
    (*x)[n] = !(*x)[n];
  }
}

/// Return the squared distance between the two points.
template<typename T, std::size_t N>
inline
T
squaredDistance(std::array<T, N> const& x, std::array<T, N> const& y)
{
  T d = 0;
  for (std::size_t n = 0; n != N; ++n) {
    d += (x[n] - y[n]) * (x[n] - y[n]);
  }
  return d;
}

/// Return the squared distance between the two points.
/** This specialization is a little faster than the dimension-general code. */
template<typename T>
inline
T
squaredDistance(std::array<T, 3> const& x, std::array<T, 3> const& y)
{
  return (x[0] - y[0]) * (x[0] - y[0]) +
         (x[1] - y[1]) * (x[1] - y[1]) +
         (x[2] - y[2]) * (x[2] - y[2]);
}


//
// Versions that use SIMD intrinsics.
//
#ifndef STLIB_NO_SIMD_INTRINSICS


#ifdef __SSE4_1__
/// Return the squared distance between the two points.
/** Specialization for single-precision, 3-D. */
inline
float
squaredDistance(std::array<float, 3> const& x, std::array<float, 3> const& y)
{
  // Take the difference of the two vectors.
  // CONTINUE: Implementent specialization for 16-byte aligned structure.
#if 0
  __m128 d = _mm_load_ps(&x[0]) - _mm_load_ps(&y[0]);
#endif
  __m128 d = _mm_set_ps(0, x[2], x[1], x[0]) -
             _mm_set_ps(0, y[2], y[1], y[0]);
  // Perform the dot product.
  d = _mm_dp_ps(d, d, 0x71);
  return *reinterpret_cast<const float*>(&d);
}
#endif


#ifdef __SSE4_1__
/// Return the squared distance between the two points.
/** Specialization for single-precision, 4-D. */
inline
float
squaredDistance(std::array<float, 4> const& x, std::array<float, 4> const& y)
{
  // Take the difference of the two vectors.
  // CONTINUE: Implementent specialization for 16-byte aligned structure.
#if 0
  __m128 d = _mm_load_ps(&x[0]) - _mm_load_ps(&y[0]);
#endif
  __m128 d = _mm_loadu_ps(&x[0]) - _mm_loadu_ps(&y[0]);
  // Perform the dot product.
  d = _mm_dp_ps(d, d, 0xF1);
  return *reinterpret_cast<const float*>(&d);
}
#endif


#ifdef __SSE4_1__
/// Return the squared distance between the two points.
/** Specialization for double-precision, 4-D. */
inline
double
squaredDistance(std::array<double, 4> const& x, std::array<double, 4> const& y)
{
  // Take the difference of the two vectors.
  // CONTINUE: Implementent specialization for 16-byte aligned structure.
#if 0
  __m128d a = _mm_load_pd(&x[0]) - _mm_load_pd(&y[0]);
  __m128d b = _mm_load_pd(&x[2]) - _mm_load_pd(&y[2]);
#endif
  __m128d a = _mm_loadu_pd(&x[0]) - _mm_loadu_pd(&y[0]);
  __m128d b = _mm_loadu_pd(&x[2]) - _mm_loadu_pd(&y[2]);
  // Perform the dot product.
  a = _mm_add_pd(_mm_dp_pd(a, a, 0x31), _mm_dp_pd(b, b, 0x31));
  return *reinterpret_cast<const double*>(&a);
}
#endif


#endif //#ifndef STLIB_NO_SIMD_INTRINSICS


/// Return the Euclidean distance between the two points.
/**
  \note distance would not be a good name for this function because
  std::distance() calculates the distance between iterators.
*/
template<typename T, std::size_t N>
inline
T
euclideanDistance(std::array<T, N> const& x, std::array<T, N> const& y)
{
  return sqrt(squaredDistance(x, y));
}

//
// Versions that use SIMD intrinsics.
//
#ifndef STLIB_NO_SIMD_INTRINSICS


// There is a slight performance penalty for using an SIMD intrinsic to
// take the square root.
#if 0
#ifdef __SSE4_1__
/// Return the Euclidean distance between the two points.
/** Specialization for single-precision, 4-D. */
inline
float
euclideanDistance(std::array<float, 4> const& x, std::array<float, 4> const& y)
{
  // Take the difference of the two vectors.
  // CONTINUE: Implement specialization for 16-byte aligned structure.
#if 0
  __m128 d = _mm_load_ps(&x[0]) - _mm_load_ps(&y[0]);
#endif
  __m128 d = _mm_loadu_ps(&x[0]) - _mm_loadu_ps(&y[0]);
  // Perform the dot product and then take the square root.
  d = _mm_sqrt_ss(_mm_dp_ps(d, d, 0xF1));
  return *reinterpret_cast<const float*>(&d);
}
#endif
#endif


#endif //#ifndef STLIB_NO_SIMD_INTRINSICS

//@}

} // namespace ext
} // namespace stlib

#define __stlib_ext_arrayStd_tcc__
#include "stlib/ext/arrayStd.tcc"
#undef __stlib_ext_arrayStd_tcc__

#endif
