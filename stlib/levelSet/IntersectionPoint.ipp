// -*- C++ -*-

#if !defined(__levelSet_IntersectionPoint_ipp__)
#error This file is an implementation detail of IntersectionPoint.
#endif

namespace stlib
{
namespace levelSet
{


// Return the signed distance to the point on the surface.
template<typename _T, std::size_t _D>
inline
_T
distance(const IntersectionPoint<_T, _D>& p, const std::array<_T, _D>& x)
{
  // Determine the sign of the distance.
  std::array<_T, _D> v = p.location;
  v -= x;
  const _T sign = dot(v, p.normal) > 0 ? 1 : -1;
  return sign * magnitude(v);
}


// If the balls intersect, calculate the points on the surface and return true.
template<typename _T>
inline
bool
makeBoundaryIntersection(const geom::Ball<_T, 2>& a, const geom::Ball<_T, 2>& b,
                         IntersectionPoint<_T, 2>* p, IntersectionPoint<_T, 2>* q)
{
  const _T d2 = ext::squaredDistance(a.center, b.center);
  if (d2 * (1. + std::numeric_limits<_T>::epsilon()) >=
      (a.radius + b.radius) * (a.radius + b.radius)) {
    return false;
  }
  // The distance between the two centers.
  const _T d = std::sqrt(d2);
  // The distance from the center of a to the supporting line of the
  // intersection points.
  assert(d > 0);
  const _T da = 0.5 * (d2 + a.radius * a.radius -
                       b.radius * b.radius) / d;
  // The distance from the supporting line of the centers to the intersection
  // points.
  const _T h = std::sqrt(a.radius * a.radius - da * da);
  // We will use the member data of p for scratch calculations before setting
  // their values. Start with the vector from the center of a to the center
  // of b.
  p->normal = b.center;
  p->normal -= a.center;
  // Normalize to a unit vector.
  ext::normalize(&p->normal);
  // The point that is the intersection of the supporting line of the centers
  // and the supporting line of the ball intersection points.
  p->location = a.center;
  p->location += da * p->normal;
  // Rotate so the vector is normal to the supporting line of the two centers.
  geom::rotatePiOver2(&p->normal);
  // Set q by translating the intersection point and taking the negative of
  // normal.
  q->location = p->location;
  std::array<_T, 2> offset = p->normal;
  offset *= h;
  q->location -= offset;
  q->normal = p->normal;
  ext::negateElements(&q->normal);
  // Set p by translating the intersection point.
  p->location += offset;

  // Compute distance up to the larger of the two radii.
  p->radius = q->radius = std::max(a.radius, b.radius);

  return true;
}


// If the three balls intersect, calculate the points on the surface and return true.
// http://en.wikipedia.org/wiki/Trilateration
template<typename _T>
inline
bool
makeBoundaryIntersection(const geom::Ball<_T, 3>& a, const geom::Ball<_T, 3>& b,
                         const geom::Ball<_T, 3>& c, IntersectionPoint<_T, 3>* p,
                         IntersectionPoint<_T, 3>* q)
{
  const std::size_t D = 3;
  typedef std::array<_T, D> Point;
  // Transform the coordinates so that the first ball is at the origin, the
  // second is at d on the x axis, and the third is at the position (i, j).
  // The basis vectors in the transformed coordinate system.
  std::array<Point, 3> e;
  // x-axis.
  e[0] = b.center - a.center;
  // In order to intersect in a circle the following inequality must hold.
  // d < a + b  and  d > |a - b|
  // In the former case the balls do not intersect. In the latter case,
  // one is inside the other.
  // If the two spheres do not intersect in a circle, return false.
  const _T d2 = ext::squaredMagnitude(e[0]);
  if (d2 * (1. + std::numeric_limits<_T>::epsilon()) >=
      (a.radius + b.radius) * (a.radius + b.radius)) {
    return false;
  }
  if (d2 * (1. - std::numeric_limits<_T>::epsilon()) <=
      (a.radius - b.radius) * (a.radius - b.radius)) {
    return false;
  }
  // Back to computing the basis vector.
  const _T d = std::sqrt(d2);
  e[0] /= d;
  const _T i = ext::dot(e[0], c.center - a.center);
  // y-axis.
  e[1] = c.center - a.center - i * e[0];
  ext::normalize(&e[1]);
  const _T j = ext::dot(e[1], c.center - a.center);
  // z-axis.
  e[2] = ext::cross(e[0], e[1]);

  // Solve for the intersection point in the transformed system
  // (x[0], x[1], +-x[2]).
  Point x;
  x[0] = (a.radius * a.radius - b.radius * b.radius + d2) / (2 * d);
  x[1] = (a.radius * a.radius - c.radius * c.radius + i * i + j * j -
          2 * i * x[0]) / (2 * j);
  const _T disc = a.radius * a.radius - x[0] * x[0] - x[1] * x[1];
  // If the ball does not intersect the circle, return false.
  if (disc < std::numeric_limits<_T>::epsilon() *
      std::numeric_limits<_T>::epsilon()) {
    return false;
  }
  x[2] = std::sqrt(disc);

  // The intersection points.
  p->location = a.center + x[0] * e[0] + x[1] * e[1] + x[2] * e[2];
  q->location = a.center + x[0] * e[0] + x[1] * e[1] - x[2] * e[2];

  // The outward normals.
  p->normal = p->location - q->location;
  ext::normalize(&p->normal);
  q->normal = - p->normal;

  // Compute distance up to the largest of the three radii.
  p->radius = q->radius = std::max(std::max(a.radius, b.radius), c.radius);

  return true;
}


// Make a bounding box that contains the points with negative distance to
// the point.
template<typename _T>
inline
void
boundNegativeDistance(const IntersectionPoint<_T, 2>& p,
                      geom::BBox<_T, 2>* box)
{
  typedef std::array<_T, 2> Point;
  // The four points of the (non-axis-aligned) box that contains the
  // half-disk with points of negative distance.
  std::array<Point, 4> corners = {{
      p.location, p.location, p.location,
      p.location
    }
  };
  // 0 1
  // 2 3
  Point offset = p.normal;
  offset *= p.radius;
  corners[2] -= offset;
  corners[3] -= offset;
  geom::rotatePiOver2(&offset);
  corners[0] += offset;
  corners[2] += offset;
  corners[1] -= offset;
  corners[3] -= offset;
  *box = geom::specificBBox<geom::BBox<_T, 2> >(corners.begin(), corners.end());
}


// Make a bounding box that contains the points with negative distance to
// the point.
template<typename _T>
inline
void
boundNegativeDistance(const IntersectionPoint<_T, 3>& p,
                      geom::BBox<_T, 3>* box)
{
  typedef std::array<_T, 3> Point;
  // The eight points of the (non-axis-aligned) box that contains the
  // points with negative distance.
  std::array<Point, 8> corners = {{
      p.location, p.location, p.location, p.location,
      p.location, p.location, p.location, p.location
    }
  };

  Point x;
  geom::computeAnOrthogonalVector(p.normal, &x);
  ext::normalize(&x);
  Point y = ext::cross(p.normal, x);
  x *= std::sqrt(_T(2)) * p.radius;
  y *= std::sqrt(_T(2)) * p.radius;

  corners[0] = p.location + x;
  corners[1] = p.location - x;
  corners[2] = p.location + y;
  corners[3] = p.location - y;

  Point offset = - p.radius * p.normal;
  for (std::size_t i = 0; i != 4; ++i) {
    corners[i + 4] = corners[i] + offset;
  }
  *box = geom::specificBBox<geom::BBox<_T, 3> >(corners.begin(), corners.end());
}


} // namespace levelSet
} // namespace stlib
