// -*- C++ -*-

/*!
  \file numerical/interpolation/simplex.h
  \brief Linear interpolation for a simplex.
*/

#if !defined(__numerical_interpolation_simplex_h__)
#define __numerical_interpolation_simplex_h__

#include "stlib/ext/array.h"

namespace stlib
{
namespace numerical
{

USING_STLIB_EXT_ARRAY_MATH_OPERATORS;

/*!
  \defgroup interpolation_simplex Linear Interpolation for a simplex.

  These functions perform linear interpolation of a field. The field is
  specified at M+1 points in N-D space. (The M+1 points form an
  M-D simplex. If M = N, then there is a unique linear function that is
  is equal to the specified field values. (For M = N = 1, two points with
  field values determine the equation of a line.)  If M < N, then the linear
  function is not uniquely determined from the specified points and field
  values. In this case, we add the constraint that the interpolating
  function is constant in directions normal to the simplex. (For M = 1
  and N = 2 the function is constrained to be constant in the direction
  orthogonal to the supporting line of the two points.)

  The general interface is:

  numerical::linear_interpolation(const std::array<std::array<T,N>,M+1>& positions,const std::array<F,M+1>& values,const std::array<T,N>& location).

  There are also interfaces that are specialized for specific dimensions.
*/

//! General interface for linear interpolation.
/*!
  \param positions are the positions of the simplex nodes. The positions
  must be distinct. (This is checked with an assertion.)
  \param values are the field values at the simplex nodes.
  \param location is the point at which to interpolate the field.

  - \c N is the dimension of the space.
  - \c M is the dimension of the simplex.
  - \c T is the number type.
  - \c F is the field type.

  \note This function is not implemented. Only specializations for certain
  values of \c N and \c M are implemented.

  \return The interpolated value of the field.

  \ingroup interpolation_linear
*/
template<std::size_t N, std::size_t M, typename T, typename F>
F
linear_interpolation(const std::array < std::array<T, N>, M + 1 > &
                     positions,
                     const std::array < F, M + 1 > & values,
                     const std::array<T, N>& location);

//
// 1-D simplex.
//

//! Specialization for a 1-D simplex in 1-D space.
/*! \ingroup interpolation_linear */
template<typename T, typename F>
F
linear_interpolation(T a, T b, const F& alpha, F beta, T x);

//! Specialization for a 1-D simplex in 1-D space.
/*! \ingroup interpolation_linear */
template<typename T, typename F>
inline
F
linear_interpolation(const std::array<T, 2>& positions,
                     const std::array<F, 2>& values,
                     const T location)
{
  return linear_interpolation(positions[0], positions[1],
                              values[0], values[1], location);
}

//! Specialization for a 1-D simplex in 2-D space.
/*! \ingroup interpolation_linear */
template<typename T, typename F>
F
linear_interpolation(const std::array<T, 2>& a,
                     const std::array<T, 2>& b,
                     const F& alpha, const F& beta,
                     const std::array<T, 2>& x);



//
// 2-D simplex.
//

//! Specialization for a 2-D simplex in 2-D space.
/*! \ingroup interpolation_linear */
template<typename T, typename F>
F
linear_interpolation(const std::array<T, 2>& a,
                     std::array<T, 2> b,
                     std::array<T, 2> c,
                     const F& alpha, F beta, F gamma,
                     std::array<T, 2> x);

//! Specialization for a 2-D simplex in 3-D space.
/*! \ingroup interpolation_linear */
template<typename T, typename F>
F
linear_interpolation(const std::array<T, 3>& a,
                     std::array<T, 3> b,
                     std::array<T, 3> c,
                     const F& alpha, F beta, F gamma,
                     std::array<T, 3>& x);



//
// 3-D simplex.
//

//! Specialization for a 3-D simplex in 3-D space.
/*! \ingroup interpolation_linear */
template<typename T, typename F>
F
linear_interpolation(const std::array<T, 3>& a,
                     std::array<T, 3> b,
                     std::array<T, 3> c,
                     std::array<T, 3> d,
                     const F& alpha, F beta,
                     F gamma, F delta,
                     std::array<T, 3> x);

} // namespace numerical
}

#define __numerical_interpolation_simplex_ipp__
#include "stlib/numerical/interpolation/simplex.ipp"
#undef __numerical_interpolation_simplex_ipp__

#endif
