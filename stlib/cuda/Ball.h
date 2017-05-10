// -*- C++ -*-

/*!
  \file cuda/Ball.h
  \brief A ball in N-dimensional space.
*/

#if !defined(__cuda_Ball_h__)
#define __cuda_Ball_h__

#include "stlib/cuda/BBox.h"
#include "stlib/cuda/limits.h"

namespace stlib
{
namespace geom
{


//! A ball in N-dimensional space.
/*!
  \param T is the number type.
  \param N is the dimension.

  A ball is defined by a center and a radius.
  This class is an aggregate type. Thus it has no user-defined constructors.
*/
template<typename T, std::size_t N>
struct Ball {
  //
  // Constants.
  //

  //! The space dimension.
  BOOST_STATIC_CONSTEXPR std::size_t Dimension = N;

  //
  // Types
  //

  //! The number type.
  typedef T Number;
  //! The representation of a point.
  typedef std::array<Number, Dimension> Point;

  //
  // Data
  //

  //! The center of the ball.
  Point center;
  //! The radius of the ball.
  Number radius;
};


//! Make a ball from the center and radius.
/*! \relates Ball */
template<typename T, std::size_t N>
inline
__device__
__host__
Ball<T, N>
makeBall(const std::array<T, N>& center, const T radius)
{
  Ball<T, N> x = {center, radius};
  return x;
}


//
// Equality Operators.
//

//! Return true if the balls are equal.
/*! \relates Ball */
template<typename T, std::size_t N>
inline
__device__
__host__
bool
operator==(const Ball<T, N>& x, const Ball<T, N>& y)
{
  return (x.center == y.center && x.radius == y.radius);
}


//! Return true if the balls are not equal.
/*! \relates Ball */
template<typename T, std::size_t N>
inline
__device__
__host__
bool
operator!=(const Ball<T, N>& x, const Ball<T, N>& y)
{
  return !(x == y);
}


//
// File I/O Operators.
//

#ifndef __CUDA_ARCH__

//! Read a ball.
/*! \relates Ball */
template<typename T, std::size_t N>
inline
std::istream&
operator>>(std::istream& in, Ball<T, N>& x)
{
  return in >> x.center >> x.radius;
}

//! Write the ball.
/*! \relates Ball */
template<typename T, std::size_t N>
inline
std::ostream&
operator<<(std::ostream& out, const Ball<T, N>& x)
{
  return out << x.center << " " << x.radius;
}

#endif

//
// Mathematical operations.
//


//! Return true if the point is inside this ball.
template<typename T, std::size_t N>
inline
__device__
__host__
bool
isInside(const Ball<T, N>& ball, const std::array<T, N>& position)
{
  return squaredDistance(ball.center, position) < ball.radius * ball.radius;
}


//! Make a bounding box around the ball.
/*! \relates Ball */
template<typename T, std::size_t N>
inline
__device__
__host__
BBox<T, N>
bound(const Ball<T, N>& ball)
{
  BBox<T, N> box = {ball.center - ball.radius,
                      ball.center + ball.radius
                     };
  return box;
}


//! Return true if the two balls intersect.
/*! \relates Ball */
template<typename T, std::size_t N>
inline
__device__
__host__
bool
doIntersect(const Ball<T, N>& a, const Ball<T, N>& b)
{
  return squaredDistance(a.center, b.center) <=
         (a.radius + b.radius) * (a.radius + b.radius);
}

//! Calculate the signed distance to the surface.
/*! \relates Ball */
template<typename T, std::size_t N>
inline
__device__
__host__
T
distance(const Ball<T, N>& ball, const std::array<T, N>& x)
{
  return euclideanDistance(ball.center, x) - ball.radius;
}

//! Calculate the closest point on the surface and return the signed distance.
/*! \relates Ball */
template<typename T, std::size_t N>
inline
__device__
__host__
T
closestPoint(const Ball<T, N>& ball, const std::array<T, N>& x,
             std::array<T, N>* closest)
{
  // Start at the point.
  *closest = x;
  // Translate to the origin.
  *closest -= ball.center;
  // Move to the surface.
  const T r = magnitude(*closest);
  // Special case that the point is at the center.
  if (r < std::numeric_limits<T>::epsilon()) {
    // Pick an arbitrary closest point.
    (*closest)[0] = ball.radius;
    for (std::size_t i = 1; i != N; ++i) {
      (*closest)[i] = 0;
    }
  }
  else {
    *closest *= (ball.radius / r);
  }
  // Translate back to the ball.
  *closest += ball.center;
  // Return the distance.
  return r - ball.radius;
}


} // namespace geom
}

#endif
