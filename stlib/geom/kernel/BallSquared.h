// -*- C++ -*-

#if !defined(__geom_kernel_BallSquared_h__)
#define __geom_kernel_BallSquared_h__

#include "stlib/ext/array.h"

#include <boost/config.hpp>

namespace stlib
{
namespace geom
{

USING_STLIB_EXT_ARRAY;

//! A ball in N-dimensional space.
/*!
  \param _T is the number type.
  \param N is the dimension.

  A ball is defined by a center and a squared radius.
  This class is an aggregate type. Thus it has no user-defined constructors.
*/
template<typename _T, std::size_t N>
struct BallSquared {
  //
  // Constants.
  //

  //! The space dimension.
  BOOST_STATIC_CONSTEXPR std::size_t Dimension = N;

  //
  // Types
  //

  //! The number type.
  typedef _T Number;
  //! The representation of a point.
  typedef std::array<Number, Dimension> Point;

  //
  // Data
  //

  //! The center of the ball.
  Point center;
  //! The squared radius of the ball.
  Number squaredRadius;
};


//! Make a ball from the center and squared radius.
/*! \relates BallSquared */
template<typename _T, std::size_t N>
inline
BallSquared<_T, N>
makeBallSquared(const std::array<_T, N>& center, const _T radius)
{
  BallSquared<_T, N> x = {center, radius};
  return x;
}


//
// Equality Operators.
//

//! Return true if the balls are equal.
/*! \relates BallSquared */
template<typename _T, std::size_t N>
inline
bool
operator==(const BallSquared<_T, N>& x, const BallSquared<_T, N>& y)
{
  return (x.center == y.center && x.squaredRadius == y.squaredRadius);
}


//! Return true if the balls are not equal.
/*! \relates BallSquared */
template<typename _T, std::size_t N>
inline
bool
operator!=(const BallSquared<_T, N>& x, const BallSquared<_T, N>& y)
{
  return !(x == y);
}


//
// File I/O Operators.
//

//! Read a ball.
/*! \relates BallSquared */
template<typename _T, std::size_t N>
inline
std::istream&
operator>>(std::istream& in, BallSquared<_T, N>& x)
{
  return in >> x.center >> x.squaredRadius;
}

//! Write the ball.
/*! \relates BallSquared */
template<typename _T, std::size_t N>
inline
std::ostream&
operator<<(std::ostream& out, const BallSquared<_T, N>& x)
{
  return out << x.center << " " << x.squaredRadius;
}


//
// Mathematical operations.
//


//! Return true if the point is inside this ball.
template<typename _T, std::size_t N>
inline
bool
isInside(const BallSquared<_T, N>& ball,
         const std::array<_T, N>& position)
{
  return squaredDistance(ball.center, position) < ball.squaredRadius;
}


} // namespace geom
}

#endif
