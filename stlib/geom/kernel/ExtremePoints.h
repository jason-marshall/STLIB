// -*- C++ -*-

/**
  \file
  \brief Implements a class for bounding objects by their extreme points.
*/

#if !defined(__geom_ExtremePoints_h__)
#define __geom_ExtremePoints_h__

#include "stlib/geom/kernel/BBox.h"

namespace stlib
{
namespace geom
{

/// Class for bounding objects by their extreme points.
/**
  \param _T is the number type.
  \param _D is the dimension.

  This class is an aggregate type. Thus it has no user-defined constructors.
*/
template<typename _T, std::size_t _D>
struct ExtremePoints {
  // Types.

  /// The number type.
  using Number = _T;
  /// The point type.
  using Point = std::array<_T, _D>;

  // Constants.

  /// The space dimension.
  BOOST_STATIC_CONSTEXPR std::size_t Dimension = _D;

  // Member data

  /// The extreme points. Index by dimension and direction bit.
  std::array<std::array<Point, 2>, Dimension> points;
};


/// Return an empty data structure, i.e., one that bounds nothing.
/** \relates ExtremePoints
    The coordinates are all NaN.
*/
template<typename _ExtremePoints>
_ExtremePoints
extremePoints();


/// Create a tight extreme points data structure for the object.
/** Specialize this struct for every object that you want to bound
    with extremePoints(). The specialization must define the 
    \c DefaultExtremePoins type. It must also have a static create()
    function that takes a const reference to the object as an argument
    and returns the extreme points data structure. */
template<typename _Float, typename _Object>
struct ExtremePointsForObject;


/// Create an extreme points data structure for the object.
/** \relates BBox */
template<typename _ExtremePoints, typename _Object>
inline
_ExtremePoints
extremePoints(_Object const& object)
{
  return ExtremePointsForObject<typename _ExtremePoints::Number, _Object>::
    create(object);
}


/// Create an extreme points data structure for the range of objects.
/** \relates ExtremePoints */
template<typename _ExtremePoints, typename _InputIterator>
inline
_ExtremePoints
extremePoints(_InputIterator begin, _InputIterator end)
{
  // If there are no objects, make the data structure empty.
  if (begin == end) {
    return extremePoints<_ExtremePoints>();
  }
  // Bound the first object.
  _ExtremePoints result = extremePoints<_ExtremePoints>(*begin++);
  // Add the rest of the objects.
  while (begin != end) {
    result += *begin++;
  }
  return result;
}


/// Create an ExtremePoints for each object in the range.
/** \relates ExtremePoints */
template<typename _ExtremePoints, typename _ForwardIterator>
inline
std::vector<_ExtremePoints>
extremePointsForEach(_ForwardIterator begin, _ForwardIterator end)
{
  std::vector<_ExtremePoints> result(std::distance(begin, end));
  for (std::size_t i = 0; i != result.size(); ++i) {
    result[i] = extremePoints<_ExtremePoints>(*begin++);
  }
  return result;
}


/// Create an empty ExtremePoints.
/** \relates ExtremePoints */
template<typename _Object>
inline
typename ExtremePointsForObject<void, _Object>::DefaultExtremePoints
defaultExtremePoints()
{
  return extremePoints<typename ExtremePointsForObject<void, _Object>::
                       DefaultExtremePoints>();
}


/// Create an ExtremePoints for the object.
/** \relates ExtremePoints */
template<typename _Object>
inline
typename ExtremePointsForObject<void, _Object>::DefaultExtremePoints
defaultExtremePoints(_Object const& object)
{
  return extremePoints<typename ExtremePointsForObject<void, _Object>::
                       DefaultExtremePoints>(object);
}


/// Create an ExtremePoints for the range of objects.
/** \relates ExtremePoints */
template<typename _InputIterator>
inline
typename ExtremePointsForObject<void, typename std::iterator_traits<_InputIterator>::value_type>::DefaultExtremePoints
defaultExtremePoints(_InputIterator begin, _InputIterator end)
{
  typedef typename ExtremePointsForObject<void, 
  typename std::iterator_traits<_InputIterator>::value_type>::
    DefaultExtremePoints ExtremePoints;
  return extremePoints<ExtremePoints>(begin, end);
}


/// Create an ExtremePoints for each object in the range.
/** \relates ExtremePoints */
template<typename _ForwardIterator>
inline
std::vector<typename ExtremePointsForObject<void, typename std::iterator_traits<_ForwardIterator>::value_type>::DefaultExtremePoints>
defaultExtremePointsForEach(_ForwardIterator begin, _ForwardIterator end)
{
  typedef typename ExtremePointsForObject<void, 
  typename std::iterator_traits<_ForwardIterator>::value_type>::
    DefaultExtremePoints ExtremePoints;
  return extremePointsForEach<ExtremePoints>(begin, end);
}


//
// Equality Operators.
//


/// Equality.
/** \relates ExtremePoints
 \note Two empty data structures are considered to be equal, even though the 
 coordinate values are not equal (because they are NaN). */
template<typename _T, std::size_t _D>
inline
bool
operator==(const ExtremePoints<_T, _D>& a, const ExtremePoints<_T, _D>& b)
{
  return (isEmpty(a) && isEmpty(b)) || a.points == b.points;
}


/// Inequality.
/** \relates ExtremePoints */
template<typename _T, std::size_t _D>
inline
bool
operator!=(const ExtremePoints<_T, _D>& a, const ExtremePoints<_T, _D>& b)
{
  return !(a == b);
}


//
// File I/O.
//


/// Read the extreme points.
/** \relates ExtremePoints */
template<typename _T, std::size_t _D>
inline
std::istream&
operator>>(std::istream& in, ExtremePoints<_T, _D>& x)
{
  return in >> x.points;
}


/// Write the extreme points box.
/** \relates ExtremePoints */
template<typename _T, std::size_t _D>
inline
std::ostream&
operator<<(std::ostream& out, const ExtremePoints<_T, _D>& x)
{
  using stlib::ext::operator<<;
  return out << x.points;
}


//
// Mathematical Functions.
//


/// Return true if no objects are being bounded.
/** An empty structure has NaN for its coordinates. */
template<typename _T, std::size_t _D>
inline
bool
isEmpty(ExtremePoints<_T, _D> const& x)
{
  // We just check the first coordinate. Note that extreme points bounding
  // structures in 0-D are empty by definition.
  return _D == 0 || x.points[0][0][0] != x.points[0][0][0];
}


/// Expand to contain the object.
template<typename _T, std::size_t _D, typename _Object>
inline
ExtremePoints<_T, _D>&
operator+=(ExtremePoints<_T, _D>& x, _Object const& rhs)
{
  return x += extremePoints<ExtremePoints<_T, _D> >(rhs);
}


/// Expand to contain the point.
template<typename _T, std::size_t _D>
ExtremePoints<_T, _D>&
operator+=(ExtremePoints<_T, _D>& x, std::array<_T, _D> const& p);


/// Expand to contain the other extreme points.
template<typename _T, std::size_t _D>
ExtremePoints<_T, _D>&
operator+=(ExtremePoints<_T, _D>& x, ExtremePoints<_T, _D> const& rhs);


/// Convert the ExtremePoints to a BBox.
template<typename _Float, typename _Float2, std::size_t _D>
struct BBoxForObject<_Float, ExtremePoints<_Float2, _D> >
{
  typedef BBox<_Float2, _D> DefaultBBox;

  static
  BBox<_Float, _D>
  create(ExtremePoints<_Float2, _D> const& x)
  {
    if (isEmpty(x)) {
      return specificBBox<BBox<_Float, _D> >();
    }
    BBox<_Float, _D> result;
    for (std::size_t i = 0; i != _D; ++i) {
      result.lower[i] = x.points[i][0][i];
      result.upper[i] = x.points[i][1][i];
    }
    return result;
  }
};


} // namespace geom
} // namespace stlib

#define __geom_ExtremePoints_ipp__
#include "stlib/geom/kernel/ExtremePoints.ipp"
#undef __geom_ExtremePoints_ipp__

#endif
