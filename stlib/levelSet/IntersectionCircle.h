// -*- C++ -*-

#if !defined(__levelSet_IntersectionCircle_h__)
#define __levelSet_IntersectionCircle_h__

#include "stlib/geom/kernel/Ball.h"
#include "stlib/geom/kernel/Circle3.h"
#include "stlib/geom/kernel/Point.h"

namespace stlib
{
namespace levelSet
{

USING_STLIB_EXT_ARRAY_MATH_OPERATORS;

//! Return the distance to the circle that is the intersection of two spheres.
/*! This function is used in negativeDistance(). It is not a
  general purpose function.  If the distance is positive, return
  infinity to indicate some unknown positive distance.  If the
  distance is negative and the closest point on the circe is not
  inside any of the intersecting balls, return the distance.
  Otherwise return negative infinity to indicate that the point is
  some unknown negative distance. */
template<typename _T>
_T
distance(const geom::Circle3<_T>& circle, const std::array<_T, 3>& x,
         const std::vector<geom::Ball<_T, 3> >& balls,
         const std::vector<std::size_t>& intersecting);


//! If the balls intersect, calculate the circle on the surface and return true.
template<typename _T>
bool
makeBoundaryIntersection(const geom::Ball<_T, 3>& a, const geom::Ball<_T, 3>& b,
                         geom::Circle3<_T>* circle);


//! Make a bounding box that contains the points with negative distance to the circle.
/*! The domain that contains the points with negative distance
 is the union of two cones with the circle as the base and the ball centers
 as the tips. We first bound this domain with the union of two pyramids.
 The base of the pyramids is a square that bounds the circle. Then we
 build a bounding box around the pyramids. */
template<typename _T>
void
boundNegativeDistance(const geom::Ball<_T, 3>& a, const geom::Ball<_T, 3>& b,
                      const geom::Circle3<_T>& circle, geom::BBox<_T, 3>* box);


} // namespace levelSet
}

#define __levelSet_IntersectionCircle_ipp__
#include "stlib/levelSet/IntersectionCircle.ipp"
#undef __levelSet_IntersectionCircle_ipp__

#endif
