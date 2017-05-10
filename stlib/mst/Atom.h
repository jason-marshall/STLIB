// -*- C++ -*-

/*!
  \file Atom.h
  \brief An atom is represented by a position and a radius.
*/

#if !defined(__mst_Atom_h__)
#define __mst_Atom_h__

#include "stlib/geom/kernel/Ball.h"

#include "stlib/geom/kernel/content.h"

namespace stlib
{
// Note that we define these functions in the geom namespace as they operate
// on geom::Ball.
namespace geom
{


//! Return true if b clips a.  c is the distance between the centers.
template<typename T>
inline
bool
doesClip(const geom::Ball<T, 3>& a, const geom::Ball<T, 3>& b, const T c)
{
  // If the two balls do not intersect.
  if (a.radius + b.radius <= c) {
    return false;
  }
  // If the first contains the second.
  if (a.radius - b.radius >= c) {
    return false;
  }
  return true;
}


//! Return true if b clips a.
template<typename T>
inline
bool
doesClip(const geom::Ball<T, 3>& a, const geom::Ball<T, 3>& b)
{
  const T c = ext::euclideanDistance(a.center, b.center);
  return doesClip(a, b, c);
}


//! Return the clipping plane distance.
template<typename T>
inline
T
computeClippingPlaneDistance(const geom::Ball<T, 3>& a,
                             const geom::Ball<T, 3>& b)
{
  // The distance between the atom's centers.
  const T c = ext::euclideanDistance(a.center, b.center);
#ifdef STLIB_DEBUG
  assert(doesClip(a, b, c));
#endif
  // The atoms should not have the same centers.
  assert(c != 0);

  // If the second atom contains the first.
  if (b.radius - a.radius >= c) {
    // The whole atom is clipped.  Return -infinity.
    return - std::numeric_limits<T>::max();
  }
  // Otherwise, the spheres intersect.

  // Compute the clipping plane distance.
  const T dist = (a.radius * a.radius - b.radius * b.radius + c * c) / (2 * c);

  // Check that the plane clips the atom.
  const T bound = ((1.0 + 10.0 * std::numeric_limits<T>::epsilon()) *
                   a.radius);
  assert(-bound <= dist && dist <= bound);

  // Return the clipping plane distance.
  return dist;
}


} // namespace geom
}

#endif
