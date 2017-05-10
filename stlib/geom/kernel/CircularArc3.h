// -*- C++ -*-

#if !defined(__geom_kernel_CircularArc3_h__)
#define __geom_kernel_CircularArc3_h__

#include "stlib/geom/kernel/Circle3.h"

#include "stlib/numerical/constants.h"

namespace stlib
{
namespace geom
{

//! A circular arc in 3-dimensional space.
/*!
  \param T is the number type.  By default it is double.

  A circular arc in 3-D is defined by a circle (acenter, a normal, and
  a radius), two
  vectors which form axes in the plane of the circle and an angle.
  The parametrization for t in [0..1] is:

  x = center + radius * (axis0 * cos(angle * t) + axis1 * sin(angle * t))

  Thus the circular arc lies on the circle.  It starts at the first axis
  and goes in the positive direction toward the second axis through the
  defined angle.
*/
template < typename T = double >
class CircularArc3
{
  //
  // Types
  //

public:

  //! The number type.
  typedef T Number;
  //! The representation of a point.
  typedef std::array<Number, 3> Point;

private:

  //! A circle.
  typedef Circle3<Number> Circle;

  //
  // Data
  //

private:

  Circle _circle;
  Point _axis0, _axis1;
  Number _angle;

public:

  //--------------------------------------------------------------------------
  //! \name Constructors etc.
  // @{

  //! Default constructor.  Uninitialized memory.
  CircularArc3() {}

  //! Construct from the center, the source, and the target.
  CircularArc3(const Point& center, const Point& source,
               const Point& target) :
    _circle(),
    _axis0(),
    _axis1(),
    _angle()
  {
    make(center, source, target);
  }

  //! Make from the center, the source, and the target.
  void
  make(const Point& center, const Point& source, const Point& target);

  // @}
  //--------------------------------------------------------------------------
  //! \name Accesors.
  // @{

  //! Return the circle.
  const Circle&
  getCircle() const
  {
    return _circle;
  }

  //! Return the center.
  const Point&
  getCenter() const
  {
    return _circle.center;
  }

  //! Return the radius.
  Number
  getRadius() const
  {
    return _circle.radius;
  }

  //! Return the first axis.
  const Point&
  getFirstAxis() const
  {
    return _axis0;
  }

  //! Return the second axis.
  const Point&
  getSecondAxis() const
  {
    return _axis1;
  }

  //! Return the angle.
  Number
  getAngle() const
  {
    return _angle;
  }

  // @}
  //--------------------------------------------------------------------------
  //! \name Functor.
  // @{

  //! Evaluate a point on the circular arc.
  /*!
    The arc is parametrized with \f$t \in [0..1]\f$.
  */
  Point
  operator()(const Number t) const
  {
    Point x = getCenter() + getRadius() * (std::cos(_angle * t) * _axis0 +
                                           std::sin(_angle * t) * _axis1);
    return x;
  }

  // @}
  //--------------------------------------------------------------------------
  //! \name Translations.
  // @{

  //! Translate by p.
  CircularArc3&
  operator+=(const Point& x)
  {
    _circle += x;
    return (*this);
  }

  //! Translate by -p.
  CircularArc3&
  operator-=(const Point& x)
  {
    _circle -= x;
    return (*this);
  }

  // @}
  //--------------------------------------------------------------------------
  //! \name Validity.
  //@{

  //! Return true if the circular arc is valid.
  bool
  isValid() const;

  //@}
};


//
// Mathematical functions.
//


//! Compute the closest point on the circular arc.
template<typename T>
void
computeClosestPoint(const CircularArc3<T>& circularArc,
                    typename CircularArc3<T>::Point x,
                    typename CircularArc3<T>::Point* closestPoint);


// As of MSVC 13, the compiler does not correctly handle default arguments.
// Apparently, this will be fixed in a future release.
// https://connect.microsoft.com/VisualStudio/feedback/details/583081/
#ifdef _MSC_VER

//! Compute the closest point on the circle to the circular arc.
template<typename T>
void
computeClosestPoint(const Circle3<T>& circle,
                    const CircularArc3<T>& circularArc,
                    typename CircularArc3<T>::Point* closestPoint,
                    T tolerance,
                    int maximumSteps);

//! Compute the closest point on the circle to the circular arc.
template<typename T>
inline
void
computeClosestPoint(const Circle3<T>& circle,
                    const CircularArc3<T>& circularArc,
                    typename CircularArc3<T>::Point* closestPoint,
                    T tolerance)
{
  computeClosestPoint(circle, circularArc, closestPoint, tolerance, 10);
}

//! Compute the closest point on the circle to the circular arc.
template<typename T>
inline
void
computeClosestPoint(const Circle3<T>& circle,
                    const CircularArc3<T>& circularArc,
                    typename CircularArc3<T>::Point* closestPoint)
{
  computeClosestPoint(circle, circularArc, closestPoint,
                      std::sqrt(std::numeric_limits<T>::epsilon()), 10);
}

#else

//! Compute the closest point on the circle to the circular arc.
template<typename T>
void
computeClosestPoint(const Circle3<T>& circle,
                    const CircularArc3<T>& circularArc,
                    typename CircularArc3<T>::Point* closestPoint,
                    T tolerance = std::sqrt(std::numeric_limits<T>::epsilon()),
                    int maximumSteps = 10);

#endif

//
// Equality Operators.
//


//! Return true if the balls are equal.
/*! \relates CircularArc3 */
template<typename T>
inline
bool
operator==(const CircularArc3<T>& x, const CircularArc3<T>& y)
{
  return (x.getCircle() == y.getCircle() &&
          x.getFirstAxis() == y.getFirstAxis() &&
          x.getSecondAxis() == y.getSecondAxis() &&
          x.getAngle() == y.getAngle());
}


//! Return true if the balls are not equal.
/*! \relates CircularArc3 */
template<typename T>
inline
bool
operator!=(const CircularArc3<T>& x, const CircularArc3<T>& y)
{
  return !(x == y);
}


//
// File I/O Operators.
//


//! Read a circular arc.
/*! \relates CircularArc3 */
template<typename T>
std::istream&
operator>>(std::istream& in, CircularArc3<T>& x);


//! Write the circular arc.
/*! \relates CircularArc3 */
template<typename T>
inline
std::ostream&
operator<<(std::ostream& out, const CircularArc3<T>& x)
{
  return out << x.getCenter() << " " << x(0.0) << " " << x(1.0);
}


} // namespace geom
}

#define __geom_kernel_CircularArc3_ipp__
#include "stlib/geom/kernel/CircularArc3.ipp"
#undef __geom_kernel_CircularArc3_ipp__

#endif
