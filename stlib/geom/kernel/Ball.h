// -*- C++ -*-

/*!
  \file
  \brief A ball in N-dimensional space.
*/

#if !defined(__geom_kernel_Ball_h__)
#define __geom_kernel_Ball_h__

#include "stlib/geom/kernel/BBox.h"

namespace stlib
{
namespace geom
{

//! A ball in N-dimensional space.
/*!
  \param _T is the number type.
  \param N is the dimension.

  A ball is defined by a center and a radius.
  This class is an aggregate type. Thus it has no user-defined constructors.
*/
template<typename _T, std::size_t N>
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
  typedef _T Number;
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


//
// Equality Operators.
//

//! Return true if the balls are equal.
/*! \relates Ball */
template<typename _T, std::size_t N>
inline
bool
operator==(const Ball<_T, N>& x, const Ball<_T, N>& y)
{
  return (x.center == y.center && x.radius == y.radius);
}


//! Return true if the balls are not equal.
/*! \relates Ball */
template<typename _T, std::size_t N>
inline
bool
operator!=(const Ball<_T, N>& x, const Ball<_T, N>& y)
{
  return !(x == y);
}


//
// File I/O Operators.
//

//! Read a ball.
/*! \relates Ball */
template<typename _T, std::size_t N>
inline
std::istream&
operator>>(std::istream& in, Ball<_T, N>& x)
{
  return in >> x.center >> x.radius;
}

//! Write the ball.
/*! \relates Ball */
template<typename _T, std::size_t N>
inline
std::ostream&
operator<<(std::ostream& out, const Ball<_T, N>& x)
{
  return out << x.center << " " << x.radius;
}

//
// Mathematical operations.
//


//! Return true if the point is inside this ball.
template<typename _T, std::size_t N>
inline
bool
isInside(const Ball<_T, N>& ball, const std::array<_T, N>& position)
{
  return ext::squaredDistance(ball.center, position) <
    ball.radius * ball.radius;
}


//! Make a bounding box around the ball.
/*! \relates BBox
  \relates Ball */
template<typename _Float, typename _Float2, std::size_t _D>
struct BBoxForObject<_Float, Ball<_Float2, _D> >
{
  typedef BBox<_Float2, _D> DefaultBBox;

  static
  BBox<_Float, _D>
  create(Ball<_Float2, _D> const& x)
  {
    return BBox<_Float, _D>{
      ext::ConvertArray<_Float>::convert(x.center - x.radius),
        ext::ConvertArray<_Float>::convert(x.center + x.radius)};
  }
};


//! Return true if the two balls intersect.
/*! \relates Ball */
template<typename _T, std::size_t N>
inline
bool
doIntersect(const Ball<_T, N>& a, const Ball<_T, N>& b)
{
  return ext::squaredDistance(a.center, b.center) <=
         (a.radius + b.radius) * (a.radius + b.radius);
}

//! Calculate the signed distance to the surface.
/*! \relates Ball */
template<typename _T, std::size_t N>
inline
_T
distance(const Ball<_T, N>& ball, const std::array<_T, N>& x)
{
  return ext::euclideanDistance(ball.center, x) - ball.radius;
}

//! Calculate the closest point on the surface and return the signed distance.
/*! \relates Ball */
template<typename _T, std::size_t N>
inline
_T
closestPoint(const Ball<_T, N>& ball, const std::array<_T, N>& x,
             std::array<_T, N>* closest)
{
  // Start at the point.
  *closest = x;
  // Translate to the origin.
  *closest -= ball.center;
  // Move to the surface.
  const _T r = ext::magnitude(*closest);
  // Special case that the point is at the center.
  if (r < std::numeric_limits<_T>::epsilon()) {
    // Pick an arbitrary closest point.
    closest->fill(0);
    (*closest)[0] = ball.radius;
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
} // namespace stlib

#endif
