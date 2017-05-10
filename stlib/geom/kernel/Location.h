// -*- C++ -*-

/**
  \file
  \brief Determine the location of an object.
*/

#if !defined(__stlib_geom_Location_h__)
#define __stlib_geom_Location_h__

#include "stlib/geom/kernel/Ball.h"

#include <utility>

namespace stlib
{
namespace geom
{


/// Get the location of an object.
/** Specialize this struct for every object that you want to get the
    location. The specialization must define the `Dimension` and the `Point`
    type. It must also have a static point() function that takes a
    const reference to the object as an argument and returns the
    location as a `Point`. */
template<typename _Object>
struct Location;


/// Get the location of the object.
/** \relates Location */
template<typename _Object>
inline
typename Location<_Object>::Point
location(_Object const& object)
{
  return Location<_Object>::point(object);
}


/// The location of a `std::array` of coordinates is the object itself.
template<typename _Float, std::size_t _D>
struct Location<std::array<_Float, _D> >
{
  /// The space dimension.
  std::size_t static constexpr Dimension = _D;
  /// A Cartesian point.
  using Point = std::array<_Float, _D>;

  /// Return the location of the point.
  /** Returning a const reference is more efficient. */
  static
  Point const&
  point(Point const& x)
  {
    return x;
  }
};


/// The location of an array of points is the centroid of the bounding box.
template<typename _Float, std::size_t _D, std::size_t N>
struct Location<std::array<std::array<_Float, _D>, N> >
{
  /// The space dimension.
  std::size_t static constexpr Dimension = _D;
  /// A Cartesian point.
  using Point = std::array<_Float, _D>;

  /// Return the centroid of the bounding box around the array of points.
  static
  Point
  point(std::array<std::array<_Float, _D>, N> const& x)
  {
    return location(bbox(x));
  }
};


/// The location of a ball is the center.
template<typename _Float, std::size_t _D>
struct Location<Ball<_Float, _D> >
{
  /// The space dimension.
  std::size_t static constexpr Dimension = _D;
  /// A Cartesian point.
  using Point = std::array<_Float, _D>;

  /// Return the center of the ball.
  /** Returning a const reference is more efficient. */
  static
  Point const&
  point(Ball<_Float, _D> const& x)
  {
    return x.center;
  }
};


/// The location of a bounding box is the centroid.
template<typename _Float, std::size_t _D>
struct Location<BBox<_Float, _D> >
{
  /// The space dimension.
  std::size_t static constexpr Dimension = _D;
  /// A Cartesian point.
  using Point = std::array<_Float, _D>;

  /// Return the centroid of the bounding box.
  static
  Point
  point(BBox<_Float, _D> const& x)
  {
    return _Float(0.5) * (x.lower + x.upper);
  }
};


/// Given a `std::pair`, we assume that `first` is the geometric object.
template<typename _Geometric, typename _Data>
struct Location<std::pair<_Geometric, _Data> >
{
  /// The space dimension.
  std::size_t static constexpr Dimension = Location<_Geometric>::Dimension;
  /// A Cartesian point.
  using Point = typename Location<_Geometric>::Point;

  /// Return the location of the first part.
  static
  Point
  point(std::pair<_Geometric, _Data> const& x)
  {
    return location(x.first);
  }
};


/// Get the location for each object in the range.
/** \relates Location */
template<typename _ForwardIterator>
inline
auto
locationForEach(_ForwardIterator begin, _ForwardIterator end) ->
  std::vector<typename Location<typename std::decay<decltype(*begin)>::type>::Point>
{
  using Point =
    typename Location<typename std::decay<decltype(*begin)>::type>::Point;
  std::vector<Point> result(std::distance(begin, end));
  for (std::size_t i = 0; i != result.size(); ++i) {
    result[i] = location(*begin++);
  }
  return result;
}


} // namespace geom
} // namespace stlib

#endif
