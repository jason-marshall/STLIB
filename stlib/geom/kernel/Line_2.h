// -*- C++ -*-

#if !defined(__geom_Line_2_h__)
#define __geom_Line_2_h__

#include <iostream>
#include <cassert>
#include <cmath>

#include "stlib/geom/kernel/SegmentMath.h"

namespace stlib
{
namespace geom
{

//! A line in 2-D.
/*!
  \param T is the number type.  By default it is double.
 */
template < typename T = double >
class Line_2
{
public:

  //
  // Public types.
  //

  //! The number type.
  typedef T Number;
  //! The segment upon which this line is built.
  typedef SegmentMath<2, T> Segment;
  //! The point type.
  typedef typename Segment::Point Point;

private:

  //
  // Member data
  //

  Segment _segment;
  Point _normal;

public:

  //--------------------------------------------------------------------------
  //! \name Constructors etc.
  // @{

  //! Default constructor.  Uninitialized memory.
  Line_2() {}

  //! Construct from points.
  Line_2(const Point& source, const Point& target) :
    _segment(source, target),
    _normal()
  {
    _normal[0] = _segment.getTangent()[1];
    _normal[1] = - _segment.getTangent()[0];
  }

  //! Make from points.
  void
  make(const Point& source, const Point& target)
  {
    _segment.make(source, target);
    _normal[0] = _segment.getTangent()[1];
    _normal[1] = - _segment.getTangent()[0];
  }

  //! Construct from a segment.
  Line_2(const Segment& segment) :
    _segment(segment),
    _normal()
  {
    _normal[0] = _segment.getTangent()[1];
    _normal[1] = - _segment.getTangent()[0];
  }

  //! Construct from a line segment.
  Line_2(const stlib::geom::Simplex<Number, 2, 1>& segment) :
    _segment(segment),
    _normal()
  {
    _normal[0] = _segment.getTangent()[1];
    _normal[1] = - _segment.getTangent()[0];
  }

  // @}
  //--------------------------------------------------------------------------
  //! \name Accesors.
  // @{

  //! Return a point on the line.
  const Point&
  getPointOn() const
  {
    return _segment.getSource();
  }

  //! Return the tangent to the line.
  const Point&
  getTangent() const
  {
    return _segment.getTangent();
  }

  //! Return the normal to the line.
  const Point&
  getNormal() const
  {
    return _normal;
  }

  //! Return the segment on which the line is built.
  const Segment&
  getSegment() const
  {
    return _segment;
  }

  // @}
  //--------------------------------------------------------------------------
  //! \name Mathematical operations.
  // @{

  //! Translate the line by \c p.
  Line_2&
  operator+=(const Point& p)
  {
    _segment += p;
    return *this;
  }

  //! Translate the line by \c -p.
  Line_2&
  operator-=(const Point& p)
  {
    _segment -= p;
    return *this;
  }

  //! Distance to the line.
  Number
  computeSignedDistance(const Point& p) const
  {
    return ext::dot(p - getPointOn(), getNormal());
  }

  //! Distance and closest point to the line.
  Number
  computeSignedDistanceAndClosestPoint(const Point& p, Point* cp) const
  {
    *cp = getPointOn() + ext::dot(p - getPointOn(),
                                  getTangent()) * getTangent();
    return ext::dot(p - getPointOn(), getNormal());
  }

  //! Compute the point where the line through \c p1 and \c p2 intersects this line.
  void
  computeIntersection(Point p1, Point p2, Point* intersectionPoint) const;

  // @}
};


//
// Mathematical Functions.
//


//! Unary positive operator.
/*! \relates Line_2 */
template<typename T>
inline
const Line_2<T>&
operator+(const Line_2<T>& x)
{
  return x;
}


//! Unary negative operator.
/*! \relates Line_2 */
template<typename T>
inline
Line_2<T>
operator-(const Line_2<T>& x)
{
  return Line_2<T>(- x.getSegment());
}


//
// File I/O.
//


//! Read a line.
/*! \relates Line_2 */
template<typename T>
inline
std::istream&
operator>>(std::istream& in, Line_2<T>& x)
{
  typename Line_2<T>::Point p, q;
  in >> p >> q;
  x = Line_2<T>(p, q);
  return in;
}


//! Write a line.
/*! \relates Line_2 */
template<typename T>
inline
std::ostream&
operator<<(std::ostream& out, const Line_2<T>& x)
{
  out << x.getSegment() << x.getNormal() << '\n';
  return out;
}


//
// Equality Operators.
//


//! Return true if the lines are equal.
/*! \relates Line_2 */
template<typename T>
inline
bool
operator==(const Line_2<T>& a, const Line_2<T>& b)
{
  return a.getSegment() == b.getSegment();
}


//! Return true if the lines are not equal.
/*! \relates Line_2 */
template<typename T>
inline
bool
operator!=(const Line_2<T>& a, const Line_2<T>& b)
{
  return !(a == b);
}


} // namespace geom
}

#define __geom_Line_2_ipp__
#include "stlib/geom/kernel/Line_2.ipp"
#undef __geom_Line_2_ipp__

#endif
