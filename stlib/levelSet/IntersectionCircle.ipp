// -*- C++ -*-

#if !defined(__levelSet_IntersectionCircle_ipp__)
#error This file is an implementation detail of negativeDistance.
#endif

namespace stlib
{
namespace levelSet
{


/*
  Return the distance to the circle that is the intersection of two spheres.

  This function is used in negativeDistance(). It is not a
  general purpose function.  If the distance is positive, return
  infinity to indicate some unknown positive distance.  If the
  distance is negative and the closest point on the circle is not
  inside any of the intersecting balls, return the distance.
  Otherwise return negative infinity to indicate that the point is
  some unknown negative distance.
*/
template<typename _T>
inline
_T
distance(const geom::Circle3<_T>& circle, const std::array<_T, 3>& x,
         const std::vector<geom::Ball<_T, 3> >& balls,
         const std::vector<std::size_t>& intersecting)
{
  // Let c be the circle center and r its radius.
  // Let a be the (signed) distance to the supporting plane of the
  // circle. Let b be the distance from the projection of the point
  // on the plane to the circle center.
  // |x - c|^2 = a^2 + b^2
  // b = sqrt(|x - circle.center|^2 - a^2)
  // The unsigned distance to the circle is d.
  // d^2 = a^2 + (b-r)^2
  // The signed distance is positive iff b > r.
  const _T sa = ext::dot(x - circle.center, circle.normal);
  const _T a2 = sa * sa;
  const _T b2 = ext::squaredDistance(x, circle.center) - a2;
  // If the distance is positive, return infinity.
  if (b2 >= circle.radius * circle.radius) {
    return std::numeric_limits<_T>::infinity();
  }

  // Compute the closest point on the circle.
  std::array<_T, 3> cp;
  computeClosestPoint(circle, x, &cp);
  // If the closest point is inside one of the intersecting spheres, return
  // -infinity.
  for (std::size_t i = 0; i != intersecting.size(); ++i) {
    if (isInside(balls[intersecting[i]], cp)) {
      return -std::numeric_limits<_T>::infinity();
    }
  }

  // Compute the distance.
  const _T b = std::sqrt(b2);
  const _T d = std::sqrt(a2 + (b - circle.radius) * (b - circle.radius));
  return -d;
}


/*
  Return the distance to the circle that is the intersection of two spheres.

  This function is used in negativeDistance(). It is not a
  general purpose function.  If the distance is positive, return
  infinity to indicate some unknown positive distance.  If the
  distance is negative and the closest point on the circle is not
  inside any of the other balls, return the distance.
  Otherwise return negative infinity to indicate that the point is
  some unknown negative distance.
*/
template<typename _T>
inline
_T
distance(const geom::Circle3<_T>& circle, const std::array<_T, 3>& x,
         const std::vector<geom::Ball<_T, 3> >& balls,
         const std::pair<std::size_t, std::size_t>& intersectionPair)
{
  // Let c be the circle center and r its radius.
  // Let a be the (signed) distance to the supporting plane of the
  // circle. Let b be the distance from the projection of the point
  // on the plane to the circle center.
  // |x - c|^2 = a^2 + b^2
  // b = sqrt(|x - circle.center|^2 - a^2)
  // The unsigned distance to the circle is d.
  // d^2 = a^2 + (b-r)^2
  // The signed distance is positive iff b > r.
  const _T sa = dot(x - circle.center, circle.normal);
  const _T a2 = sa * sa;
  const _T b2 = squaredDistance(x, circle.center) - a2;
  // If the distance is positive, return infinity.
  if (b2 >= circle.radius * circle.radius) {
    return std::numeric_limits<_T>::infinity();
  }

  // Compute the closest point on the circle.
  std::array<_T, 3> cp;
  computeClosestPoint(circle, x, &cp);
  // If the closest point is inside one of the other spheres, return
  // -infinity.
  for (std::size_t i = 0; i != balls.size(); ++i) {
    if (i == intersectionPair.first || i == intersectionPair.second) {
      continue;
    }
    if (isInside(balls[i], cp)) {
      return -std::numeric_limits<_T>::infinity();
    }
  }

  // Compute the distance.
  const _T b = std::sqrt(b2);
  const _T d = std::sqrt(a2 + (b - circle.radius) * (b - circle.radius));
  return -d;
}


// If the balls intersect, calculate the circle on the surface and return true.
template<typename _T>
inline
bool
makeBoundaryIntersection(const geom::Ball<_T, 3>& a, const geom::Ball<_T, 3>& b,
                         geom::Circle3<_T>* circle)
{
  // To abuse notation, let a and b denote the radii of the balls.
  // Let c be the distance between the ball centers.
  const _T c2 = ext::squaredDistance(a.center, b.center);
  if (c2 * (1. + std::numeric_limits<_T>::epsilon()) >=
      (a.radius + b.radius) * (a.radius + b.radius)) {
    return false;
  }

  // From the two ball centers, we calculate the normal to the supporting
  // plane of the circle.
  circle->normal = b.center;
  circle->normal -= a.center;
  ext::normalize(&circle->normal);

  // The distance between the two centers.
  const _T c = std::sqrt(c2);
  if (c <= 0) {
    return false;
  }
  // Consider the triangle with sides a, b, and c.
  // By the law of cosines:
  //   b^2 = a^2 + c^2 - 2 a c cos(beta)
  //   cos(beta) = (a^2 + c^2 - b^2)/(2 a c)
  // Let da be the distance from the center of a to the supporting plane
  // of the circle. For the definition of the cosine:
  //   cos(beta) = da / a.
  // From this we can calculate da.
  //   da = a cos(beta)
  //   da = (a^2 + c^2 - b^2)/(2 c)
  const _T da = 0.5 * (c2 + a.radius * a.radius -
                       b.radius * b.radius) / c;
  // From this we calculate the circle center.
  circle->center = a.center;
  circle->center += da * circle->normal;

  // The squared height of the triangle.
  const _T h2 = a.radius * a.radius - da * da;
  if (h2 <= 0) {
    return false;
  }
  // The radius of the circle is the height of the triangle.
  circle->radius = std::sqrt(h2);

  return true;
}


// Make a bounding box that contains the points with negative distance to
// the circle. The domain that contains the points with negative distance
// is the union of two cones with the circle as the base and the ball centers
// as the tips. We first bound this domain with the union of two pyramids.
// The base of the pyramids is a square that bounds the circle. Then we
// build a bounding box around the pyramids.
template<typename _T>
inline
void
boundNegativeDistance(const geom::Ball<_T, 3>& a, const geom::Ball<_T, 3>& b,
                      const geom::Circle3<_T>& circle, geom::BBox<_T, 3>* box)
{
  typedef std::array<_T, 3> Point;
  // Basis vectors for the supporting plane of the circle.
  Point x, y;
  geom::computeAnOrthogonalVector(circle.normal, &x);
  ext::normalize(&x);
  y = ext::cross(circle.normal, x);
  const _T length = std::sqrt(_T(2.)) * circle.radius;
  x *= length;
  y *= length;
  // The six corners of the union of the pyramids. Four points for the
  // shared base. Two points for the tops.
  std::array<Point, 6> corners = {{
      circle.center - x,
      circle.center + x,
      circle.center - y,
      circle.center + y,
      a.center,
      b.center
    }
  };
  *box = geom::specificBBox<geom::BBox<_T, 3> >(corners.begin(), corners.end());
}


} // namespace levelSet
}
