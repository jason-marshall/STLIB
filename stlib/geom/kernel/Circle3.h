// -*- C++ -*-

#if !defined(__geom_kernel_Circle3_h__)
#define __geom_kernel_Circle3_h__

#include "stlib/geom/kernel/Point.h"
#include "stlib/geom/kernel/SegmentMath.h"

#include <limits>

namespace stlib
{
namespace geom
{

//! A circle in 3-dimensional space.
/*!
  \param _T is the number type.

  A circle in 3-D is defined by a center, a normal, and a radius.
*/
template<typename _T>
struct Circle3 {
  //
  // Types
  //

  //! The number type.
  typedef _T Number;
  //! The representation of a point.
  typedef std::array<Number, 3> Point;

  //
  // Data
  //

  Point center;
  Point normal;
  Number radius;

  //--------------------------------------------------------------------------
  //! \name Translations.
  // @{

  //! Translate by p.
  Circle3&
  operator+=(const Point& x)
  {
    center += x;
    return (*this);
  }

  //! Translate by -p.
  Circle3&
  operator-=(const Point& x)
  {
    center -= x;
    return (*this);
  }

  // @}
  //--------------------------------------------------------------------------
  //! \name Validity.
  //@{

  //! Return true if the circle is valid.
  bool
  isValid() const;

  //@}
};

//
// Mathematical functions.
//

//! Compute the closest point on the circle.
template<typename _T>
void
computeClosestPoint(const Circle3<_T>& circle,
                    typename Circle3<_T>::Point x,
                    typename Circle3<_T>::Point* closestPoint);

// As of MSVC 13, the compiler does not correctly handle default arguments.
// Apparently, this will be fixed in a future release.
// https://connect.microsoft.com/VisualStudio/feedback/details/583081/
#ifdef _MSC_VER

//! Compute the closest point on the circle to the edge.
template<typename _T>
void
computeClosestPoint
(const Circle3<_T>& circle,
 const typename Circle3<_T>::Point& source,
 const typename Circle3<_T>::Point& target,
 typename Circle3<_T>::Point* closestPoint,
 _T tolerance,
 std::size_t maximumSteps);

//! Compute the closest point on the circle to the edge.
template<typename _T>
inline
void
computeClosestPoint
(const Circle3<_T>& circle,
 const typename Circle3<_T>::Point& source,
 const typename Circle3<_T>::Point& target,
 typename Circle3<_T>::Point* closestPoint,
 _T tolerance)
{
  computeClosestPoint(circle, source, target, closestPoint, tolerance, 10);
}

//! Compute the closest point on the circle to the edge.
template<typename _T>
inline
void
computeClosestPoint
(const Circle3<_T>& circle,
 const typename Circle3<_T>::Point& source,
 const typename Circle3<_T>::Point& target,
 typename Circle3<_T>::Point* closestPoint)
{
  computeClosestPoint(circle, source, target, closestPoint, 
                      std::sqrt(std::numeric_limits<_T>::epsilon()), 10);
}

#else

//! Compute the closest point on the circle to the edge.
template<typename _T>
void
computeClosestPoint
(const Circle3<_T>& circle,
 const typename Circle3<_T>::Point& source,
 const typename Circle3<_T>::Point& target,
 typename Circle3<_T>::Point* closestPoint,
 _T tolerance = std::sqrt(std::numeric_limits<_T>::epsilon()),
 std::size_t maximumSteps = 10);

#endif

//
// Equality Operators.
//

//! Return true if the balls are equal.
/*! \relates Circle3 */
template<typename _T>
inline
bool
operator==(const Circle3<_T>& x, const Circle3<_T>& y)
{
  return (x.center == y.center && x.normal == y.normal &&
          x.radius == y.radius);
}


//! Return true if the balls are not equal.
/*! \relates Circle3 */
template<typename _T>
inline
bool
operator!=(const Circle3<_T>& x, const Circle3<_T>& y)
{
  return !(x == y);
}


//
// File I/O Operators.
//

//! Read a circle.
/*! \relates Circle3 */
template<typename _T>
inline
std::istream&
operator>>(std::istream& in, Circle3<_T>& x)
{
  return in >> x.center >> x.normal >> x.radius;
}

//! Write the circle.
/*! \relates Circle3 */
template<typename _T>
inline
std::ostream&
operator<<(std::ostream& out, const Circle3<_T>& x)
{
  return out << x.center << " " << x.normal << " " << x.radius;
}


} // namespace geom
}

#define __geom_kernel_Circle3_ipp__
#include "stlib/geom/kernel/Circle3.ipp"
#undef __geom_kernel_Circle3_ipp__

#endif
