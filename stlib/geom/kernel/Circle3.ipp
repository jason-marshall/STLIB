// -*- C++ -*-

#if !defined(__geom_kernel_Circle3_ipp__)
#error This file is an implementation detail of the class Circle3.
#endif

namespace stlib
{
namespace geom
{


//
// Validity.
//


// Return true if the circle is valid.
template<typename _T>
inline
bool
Circle3<_T>::
isValid() const
{
  // If the radius is negative.
  if (radius < 0) {
    // The circle is not valid.
    return false;
  }
  // If the normal is not of unit length.
  if (std::abs(ext::magnitude(normal) - 1.0) >
      10.0 * std::numeric_limits<_T>::epsilon()) {
    // The circle is not valid.
    return false;
  }
  // Otherwise, the circle is valid.
  return true;
}


//
// Mathematical functions.
//


// Compute the closest point on the circle.
template<typename _T>
inline
void
computeClosestPoint(const Circle3<_T>& circle,
                    typename Circle3<_T>::Point x,
                    typename Circle3<_T>::Point* closestPoint)
{
  typedef typename Circle3<_T>::Point Point;

  // The vector between the point and the circle center.
  Point vec = x;
  vec -= circle.center;
  // Move the point into the plane of the circle.
  x -= ext::dot(vec, circle.normal) * circle.normal;
  // The vector between the point in the plane and the circle center.
  vec = x;
  vec -= circle.center;

  // Deal with vec near zero length.
  if (ext::magnitude(vec) < 10.0 * std::numeric_limits<_T>::epsilon()) {
    computeAnOrthogonalVector(circle.normal, &vec);
  }

  // Change the vector length to the circle radius.
  ext::normalize(&vec);
  vec *= circle.radius;
  // The closest point is center + vec.
  *closestPoint = circle.center;
  *closestPoint += vec;
}


// Compute the closest point on the circle to the edge.
template<typename _T>
inline
void
computeClosestPoint(const Circle3<_T>& circle,
                    const typename Circle3<_T>::Point& source,
                    const typename Circle3<_T>::Point& target,
                    typename Circle3<_T>::Point* closestPoint,
                    _T tolerance, std::size_t maximumSteps)
{
  typedef typename Circle3<_T>::Point Point;

  // Make a line segment from the endpoints.
  SegmentMath<3, _T> segment(source, target);

  // Start with the mid-point of the line segment.
  Point pointOnSegment = source;
  pointOnSegment = target;
  pointOnSegment *= 0.5;

  // Compute an initial closest point on the circle.
  computeClosestPoint(circle, pointOnSegment, closestPoint);

  // Iterate computing the closest point until we achieve convergence.
  Point pointOnCircle;
  // We have taken one step so far.
  std::size_t numberOfSteps = 1;
  do {
    // Increment the number of steps.
    ++numberOfSteps;
    // Record the old point on the circle.
    pointOnCircle = *closestPoint;
    // Compute the closest point on the line segment.
    computeClosestPoint(segment, pointOnCircle, &pointOnSegment);
    // Compute the closest point on the circle to the point on the segment.
    computeClosestPoint(circle, pointOnSegment, closestPoint);
  }
  while (numberOfSteps < maximumSteps &&
         ext::euclideanDistance(pointOnCircle, *closestPoint) * circle.radius
         > tolerance);
}


} // namespace geom
}
