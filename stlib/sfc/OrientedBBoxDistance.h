// -*- C++ -*-

#if !defined(__sfc_OrientedBBoxDistance_h__)
#define __sfc_OrientedBBoxDistance_h__

/*!
  \file
  \brief Compute lower bounds on the distance to an oriented bounding box.
*/

#include "stlib/geom/kernel/OrientedBBox.h"

namespace stlib
{
namespace sfc
{

USING_STLIB_EXT_ARRAY_MATH_OPERATORS;

//! Compute lower bounds on the distance to an oriented bounding box.
/*!
  \param _Float is the floating-point number type.
  \param _D is the dimension.

  Consider a point outside a bounding box (either axis-aligned or oriented).
  One can obtain a lower bound on the distance to the contained objects 
  by bounding the distance to the surface of the box. A lower bound on 
  the squared distance is the minimum squared distance to a corner minus
  <em>r<sup>2</sup></em> where <em>r</em> is the radius of the largest 
  face.

  This class is used to determine if the objects contained in the
  oriented bounding box are relevant to a group of points. It is
  relevant if any of the lower bounds on the distance are less than or
  equal to the upper bounds. The upper bounds must be supplied. This
  algorithm provides a more accurate bound than the distance to an
  axis-aligned bounding box, but less accurate (and less expensive)
  than the distance to an oriented bounding box.  (Computing the
  distance to an OBB is relatively expensive because of the requisite
  change of basis for the query points.) The algorithm is specialized
  for the case that the objects lie close to a plane, specifically,
  the distance to the plane is small compared to the total extents of
  the objects. The plane that passes through the center of the box and
  is normal to the direction with minimum radius is one
  candidate. Note that the largest faces of the box are parallel to
  this plane. We only apply the algorithm if all of the query points
  are above one of these two largest faces. (Note that this is a
  sufficient condition for all of the points being outside the box.)
  When testing this requirement, we only compute the distance from the
  center of the query points bounding box to the plane. Then we use an
  upper bound on the size of the query points bounding box and the
  radius of the oriented bounding box to determine if the algorithm
  may be applied.
*/
template<typename _Float, std::size_t _D>
class OrientedBBoxDistance
{
public:

  //! The space dimension.
  BOOST_STATIC_CONSTEXPR std::size_t Dimension = _D;
  //! A Cartesian point.
  typedef std::array<_Float, Dimension> Point;
  //! An oriented bounding box.
  typedef geom::OrientedBBox<_Float, Dimension> OrientedBBox;

private:

  //! The number of corners on the box.
  BOOST_STATIC_CONSTEXPR std::size_t NumFaceCorners =
    std::size_t(1) << (Dimension - 1);
  //! The SIMD vector type.
  typedef typename simd::Vector<_Float>::Type Vector;
  //! The SIMD vector size.
  BOOST_STATIC_CONSTEXPR std::size_t VectorSize = simd::Vector<_Float>::Size;

  //! The oriented bounding box.
  OrientedBBox _orientedBBox;
  //! The corners for the two largest faces.
  std::array<std::array<Point, NumFaceCorners>, 2> _corners;
  //! The squared radius of the largest face.
  _Float _faceRadius2;

public:

  //! Construct from an oriented bounding box.
  OrientedBBoxDistance(OrientedBBox const& orientedBBox);

  //! Default constructor results in uninitialized data.
  OrientedBBoxDistance() :
    _orientedBBox(),
    _corners(),
    _faceRadius2()
  {
  }

  //! Return the oriented bounding box.
  OrientedBBox const&
  orientedBBox() const
  {
    return _orientedBBox;
  }

  //! If the lower bound can be applied, return a direction. Otherwise return -1.
  std::size_t
  getDirection(Point const& queryPointsCenter, _Float queryPointsMaxRadius)
    const;

  //! Return true if the contained objects are relevant to any of the query points.
  bool
areAnyRelevant
  (Point queryPointsCenter,
   _Float queryPointsMaxRadius,
   std::size_t direction, 
   std::vector<_Float, simd::allocator<_Float> > const& queryPointData,
   std::vector<_Float, simd::allocator<_Float> > const& upperBounds) const;
};

} // namespace sfc
}

#define __sfc_OrientedBBoxDistance_tcc__
#include "stlib/sfc/OrientedBBoxDistance.tcc"
#undef __sfc_OrientedBBoxDistance_tcc__

#endif
