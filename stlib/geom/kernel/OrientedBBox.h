// -*- C++ -*-

/*!
  \file
  \brief Implements a class for an oriented bounding box in N dimensions.
*/

#if !defined(__geom_OrientedBBox_h__)
#define __geom_OrientedBBox_h__

#include "stlib/geom/kernel/BBox.h"
#include "stlib/simd/allocator.h"
#include "stlib/simd/functions.h"

#include "Eigen/SVD"

#include <vector>

namespace stlib
{
namespace geom
{

//! An oriented bounding box in the specified dimension.
/*!
  \param _Float is the floating-point number type.
  \param _D is the dimension.

  This class is an aggregate type. Thus it has no user-defined constructors.
*/
template<typename _Float, std::size_t _D>
struct OrientedBBox {
  //
  // Types.
  //

  //! The number type.
  typedef _Float Number;
  //! The point type.
  typedef std::array<_Float, _D> Point;

  //
  // Constants.
  //

  //! The space dimension.
  BOOST_STATIC_CONSTEXPR std::size_t Dimension = _D;

  //
  // Member data
  //

  //! The center of the box.
  Point center;
  //! The orthonormal axes.
  std::array<Point, Dimension> axes;
  //! The radii.
  Point radii;

  //
  // Member functions.
  //

  //! Make an oriented bounding box using principal components analysis.
  void
  buildPca(std::vector<Point> const& points);

  //! Make an oriented bounding box using principal components analysis.
  /*!
   \param begin The beginning of a sequence of points.
   \param end One past the end of a sequence of points.
  */
  template<typename _ForwardIterator>
  void
  buildPca(_ForwardIterator begin, _ForwardIterator end);

  //! Make an oriented bounding box using principal components analysis.
  /*! Try combining the first and second principal directions to obtain a 
    box with smaller volume. */
  void
  buildPcaRotate(std::vector<Point> const& points);

  //! Build the associated origin-centered, axis-aligned bounding box.
  BBox<_Float, _D>
  bbox() const
  {
    return BBox<_Float, _D>{-radii, radii};
  }

  //! Transform the point into the coordinates for the associated AABB.
  Point
  transform(Point p) const;

  //! Transform the points into the coordinates for the associated AABB.
  void
  transform(std::vector<_Float, simd::allocator<_Float> > const& input,
            std::vector<_Float, simd::allocator<_Float> >* output) const;

private:

  //! Update the center and calculate the radii using the axes.
  void
  _updateCenterCalculateRadii(std::vector<Point> const& points);
};


//
// Equality Operators.
//


//! Equality.
/*! \relates OrientedBBox */
template<typename _Float, std::size_t _D>
inline
bool
operator==(const OrientedBBox<_Float, _D>& a, const OrientedBBox<_Float, _D>& b)
{
  return (a.center == b.center && a.axes == b.axes && a.radii == b.radii);
}


//! Inequality.
/*! \relates OrientedBBox */
template<typename _Float, std::size_t _D>
inline
bool
operator!=(const OrientedBBox<_Float, _D>& a, const OrientedBBox<_Float, _D>& b)
{
  return !(a == b);
}


//
// File I/O.
//


//! Read the bounding box.
/*! \relates OrientedBBox */
template<typename _Float, std::size_t _D>
inline
std::istream&
operator>>(std::istream& in, OrientedBBox<_Float, _D>& x)
{
  return in >> x.center >> x.axes >> x.radii;
}


//! Write the bounding box.
/*! \relates OrientedBBox */
template<typename _Float, std::size_t _D>
inline
std::ostream&
operator<<(std::ostream& out, const OrientedBBox<_Float, _D>& x)
{
  return out << x.center << ' ' << x.axes << ' ' << x.radii;
}


//
// Mathematical Functions.
//


//! Return the content (length, area, volume, etc.) of the box.
template<typename _Float, std::size_t _D>
inline
_Float
content(OrientedBBox<_Float, _D> const& x)
{
  _Float c = 1;
  for (std::size_t i = 0; i != _D; ++i) {
    c *= 2 * x.radii[i];
  }
  return c;
}

} // namespace geom
}

#define __geom_OrientedBBox_ipp__
#include "stlib/geom/kernel/OrientedBBox.ipp"
#undef __geom_OrientedBBox_ipp__

#endif
