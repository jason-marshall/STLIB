// -*- C++ -*-

#if !defined(__geom_kernel_Ball_h__)
#define __geom_kernel_Ball_h__

#include "stlib/ext/array.h"

namespace stlib
{
namespace mst
{

// CONTINUE Get rid of this class.
//! A ball in N dimensional space.
/*!
  \param N is the dimension.
  \param T is the number type.  By default it is double.

  A ball is defined by a center and a radius.
*/
template<std::size_t N, typename T = double>
class Ball
{
  //
  // Types
  //

public:

  //! The number type.
  typedef T Number;
  //! The representation of a point.
  typedef std::array<Number, N> Point;

  //
  // Data
  //

private:

  Point _center;
  Number _radius;

public:

  //--------------------------------------------------------------------------
  /*! \name Constructors etc.
    We use the default copy constructor, assignment operator, and destructor.
  */
  // @{

  //! Default constructor.  Uninitialized memory.
  Ball() {}

  //! Construct from a center and a radius.
  Ball(const Point& center, const Number radius) :
    _center(center),
    _radius(radius) {}

  //! Make from a center and radius.
  void
  make(const Point& center, const Number radius)
  {
    _center = center;
    _radius = radius;
  }

  // @}
  //--------------------------------------------------------------------------
  //! \name Accesors.
  // @{

  //! Return the center.
  const Point&
  center() const
  {
    return _center;
  }

  //! Return the radius.
  Number
  radius() const
  {
    return _radius;
  }

  // @}
  //--------------------------------------------------------------------------
  //! \name Manipulators.
  // @{

  //! Return a reference to the center.
  Point&
  center()
  {
    return _center;
  }

  //! Return a reference to the radius.
  Number&
  radius()
  {
    return _radius;
  }

  // @}
  //--------------------------------------------------------------------------
  //! \name Translations.
  // @{

  //! Translate by p.
  Ball&
  operator+=(const Point& x)
  {
    _center += x;
    return (*this);
  }

  //! Translate by -p.
  Ball&
  operator-=(const Point& x)
  {
    _center -= x;
    return (*this);
  }

  // @}
  //--------------------------------------------------------------------------
  //! \name Mathematical functions.
  //@{

  //! Return true if the point is inside this ball.
  bool
  isInside(const Point& position) const
  {
    return squaredDistance(_center, position) < _radius * _radius;
  }

  //@}
};


//
// Equality Operators.
//

//! Return true if the balls are equal.
/*! \relates Ball */
template<std::size_t N, typename T>
inline
bool
operator==(const Ball<N, T>& x, const Ball<N, T>& y)
{
  return (x.center() == y.center() && x.radius() == y.radius());
}


//! Return true if the balls are not equal.
/*! \relates Ball */
template<std::size_t N, typename T>
inline
bool
operator!=(const Ball<N, T>& x, const Ball<N, T>& y)
{
  return !(x == y);
}


//
// File I/O Operators.
//

//! Read a ball.
/*! \relates Ball */
template<std::size_t N, typename T>
inline
std::istream&
operator>>(std::istream& in, Ball<N, T>& x)
{
  return in >> x.center() >> x.radius();
}

//! Write the ball.
/*! \relates Ball */
template<std::size_t N, typename T>
inline
std::ostream&
operator<<(std::ostream& out, const Ball<N, T>& x)
{
  return out << x.center() << " " << x.radius();
}

} // namespace mst
}

#endif
