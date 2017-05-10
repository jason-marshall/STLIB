// -*- C++ -*-

/**
  \file
  \brief A segment in N dimensional space designed for doing math operations.
*/

#if !defined(__geom_SegmentMath_h__)
#define __geom_SegmentMath_h__

#include "stlib/geom/kernel/Simplex.h"
#include "stlib/geom/kernel/content.h"

#include <iosfwd>
#include <limits>

#include <cmath>

namespace stlib
{
namespace geom
{

/// A segment in N dimensional space designed for doing math operations.
/**
  \param N is the dimension.
  \param T is the number type. By default it is double.

  A segment is an ordered pair of points. This class stores the length
  of the segment and its tangent.
*/
template < std::size_t N, typename T = double >
class SegmentMath
{
public:

  /// The number type.
  typedef T Number;
  /// The representation of a point.
  typedef std::array<T, N> Point;

private:

  Simplex<Number, N, 1> _segment;
  Point _tangent;
  Number _length;

public:

  //--------------------------------------------------------------------------
  /** \name Constructors etc.
    Use the synthesized copy constructor, assignment operator, and destructor.
  */
  // @{

  /// Default constructor. Uninitialized memory.
  SegmentMath() :
    _segment(),
    _tangent(),
    _length() {}

  /// Construct from two points.
  SegmentMath(Point const& source, Point const& target)
  {
    make(source, target);
  }

  /// Construct from a Segment
  SegmentMath(Simplex<T, N, 1> const& s)
  {
    make(s[0], s[1]);
  }

  /// Make from two points.
  void
  make(Point const& source, Point const& target);

  // @}
  //--------------------------------------------------------------------------
  /// \name Accesors.
  // @{

  /// Return the first point of the line segment.
  Point const&
  getSource() const
  {
    return _segment[0];
  }

  /// Return the second point of the line segment.
  Point const&
  getTarget() const
  {
    return _segment[1];
  }

  /// Return the unit tangent to the line segment.
  Point const&
  getTangent() const
  {
    return _tangent;
  }

  /// Return the length of the line segment.
  T
  getLength() const
  {
    return _length;
  }

  // @}
  //--------------------------------------------------------------------------
  /// \name Translations.
  // @{

  /// Translate by p.
  SegmentMath&
  operator+=(Point const& p);

  /// Translate by -p.
  SegmentMath&
  operator-=(Point const& p);

  // @}
  //--------------------------------------------------------------------------
  /// \name Validity.
  // @{

  /// Return true if the segment is valid.
  bool
  isValid() const;

  // @}
};


//
// Distance.
//

//
// Unary Operators.
//


/// Return the segment.
/** \relates SegmentMath */
template<std::size_t N, typename T>
inline
SegmentMath<N, T> const&
operator+(SegmentMath<N, T> const& x)
{
  return x;
}


/// Return a reversed segment.
/** \relates SegmentMath */
template<std::size_t N, typename T>
inline
SegmentMath<N, T>
operator-(SegmentMath<N, T> const& x)
{
  return SegmentMath<N, T>(x.getTarget(), x.getSource());
}


//
// Equality Operators.
//


/// Return true if the segments are equal.
/** \relates SegmentMath */
template<std::size_t N, typename T>
inline
bool
operator==(SegmentMath<N, T> const& x, SegmentMath<N, T> const& y)
{
  return (x.getSource() == x.getSource() &&
          x.getTarget() == x.getTarget() &&
          x.getTangent() == y.getTangent() && x.getLength() == y.getLength());
}


/// Return true if the segments are not equal.
/** \relates SegmentMath */
template<std::size_t N, typename T>
inline
bool
operator!=(SegmentMath<N, T> const& x, SegmentMath<N, T> const& y)
{
  return !(x == y);
}


//
// Mathematical Functions.
//


/// Compute the unsigned distance to the line segment.
/** \relates SegmentMath */
template<std::size_t N, typename T>
T
computeDistance(SegmentMath<N, T> const& segment,
                const typename SegmentMath<N, T>::Point& x);


/// Compute closest point on the line segment.
/** \relates SegmentMath */
template<std::size_t N, typename T>
void
computeClosestPoint(SegmentMath<N, T> const& segment,
                    const typename SegmentMath<N, T>::Point& x,
                    typename SegmentMath<N, T>::Point* closestPoint);


/// Compute the unsigned distance to the line segment and the closest point on it.
/** \relates SegmentMath */
template<std::size_t N, typename T>
T
computeDistanceAndClosestPoint(SegmentMath<N, T> const& segment,
                               const typename SegmentMath<N, T>::Point& x,
                               typename SegmentMath<N, T>::Point* closestPoint);


/// Compute the unsigned distance to the supporting line of the segment.
/** \relates SegmentMath */
template<std::size_t N, typename T>
T
computeUnsignedDistanceToSupportingLine
(SegmentMath<N, T> const& segment,
 const typename SegmentMath<N, T>::Point& x);


/// Compute the unsigned distance to the supporting line of the segment and the closest point on the line.
/** \relates SegmentMath */
template<std::size_t N, typename T>
T
computeUnsignedDistanceAndClosestPointToSupportingLine
(SegmentMath<N, T> const& segment, const typename SegmentMath<N, T>::Point& x,
 typename SegmentMath<N, T>::Point* closestPoint);


/// Return true if the segment intersects the plane of constant z.
/**
  \relates SegmentMath
  Set x and y to the point of intersection.
*/
template<typename T>
bool
computeZIntersection(const SegmentMath<3, T>& segment, T* x, T* y, T z);

/// If the two Segments intersect, return true and the point of intersection. Otherwise return false.
/** \relates SegmentMath */
template<typename T>
bool
computeIntersection(const SegmentMath<2, T>& s1, const SegmentMath<2, T>& s2,
                    typename SegmentMath<2, T>::Point* intersectionPoint);


//
// File I/O Operators.
//


/// File input.
/** \relates SegmentMath */
template<std::size_t N, typename T>
std::istream&
operator>>(std::istream& in, SegmentMath<N, T>& s);


/// File output.
/** \relates SegmentMath */
template<std::size_t N, typename T>
std::ostream&
operator<<(std::ostream& out, SegmentMath<N, T> const& s);


} // namespace geom
} // namespace stlib

#define __geom_SegmentMath_ipp__
#include "stlib/geom/kernel/SegmentMath.ipp"
#undef __geom_SegmentMath_ipp__

#endif
