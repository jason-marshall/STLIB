// -*- C++ -*-

#if !defined(__geom_Line_2_ipp__)
#error This file is an implementation detail of the class Line_2.
#endif

namespace stlib
{
namespace geom
{

//
// Mathematical functions
//


template<typename T>
inline
void
Line_2<T>::
computeIntersection(Point q1, Point q2, Point* intersectionPoint) const
{
#ifdef STLIB_DEBUG
  assert(computeSignedDistance(q1) * computeSignedDistance(q2) <= 0);
#endif
  q1 -= getPointOn();
  q2 -= getPointOn();

  const Number p1 = ext::dot(q1, getTangent());
  const Number p2 = ext::dot(q2, getTangent());
  const Number h1 = ext::dot(q1, getNormal());
  const Number h2 = ext::dot(q2, getNormal());

  *intersectionPoint = getPointOn() +
                       ((p1 * h2 - p2 * h1) / (h2 - h1)) * getTangent();
}

} // namespace geom
} // namespace stlib
