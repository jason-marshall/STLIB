// -*- C++ -*-

/*!
  \file triangulateAtomRubber.h
  \brief Triangulate with rubber clipping.
*/

#if !defined(__mst_triangulateAtomRubber_h__)
#define __mst_triangulateAtomRubber_h__

#include "stlib/mst/tesselate_sphere.h"
#include "stlib/mst/Atom.h"

#include "stlib/ads/algorithm/sort.h"
#include "stlib/geom/kernel/Circle3.h"
#include "stlib/geom/kernel/CircularArc3.h"
#include "stlib/geom/mesh/iss/accessors.h"

namespace stlib
{
namespace mst
{


//! Triangulate the visible surface.
/*!
  \param atom The atom to triangulate.
  \param clippingIdentifiers The identifiers of the atoms that might
  clip the surface.
  \param clippingAtoms The atoms that might clip the surface.
  \param edgeLengthSlope The slope of the maximum edge length function.
  \param edgeLengthOffset The offset of the maximum edge length function.
  \param refinementLevel The refinement level for the initial mesh.
  (Used only if it is non-negative.)
  \param mesh The output triangle mesh.
  \param actuallyClip The identifiers of the atoms that actually clip the mesh.
  \param maximumStretchFactor Let x be the mean edge length in the initial
  tesselation of the atom.   A clipping operation will be performed only if
  the resulting edge lengths are between x * maximumStretchFactor and
  x / maximumStretchFactor.  The value of this parameter must be between
  0 and 1.  If the maximum stretch factor is zero (the default value) then
  all clipping operations are allowed.  If it is 1, no clipping operations
  are allowed.
  \param areUsingCircularEdges specifies whether the edges should be
  interpreted as arcs of a great circle on the atomic surface.

  \return The target edge length.
*/
template<typename T, typename IntOutputIterator>
T
triangulateVisibleSurfaceWithRubberClipping
(const geom::Ball<T, 3>& atom,
 std::vector<std::size_t>& clippingIdentifiers,
 std::vector< geom::Ball<T, 3> >& clippingAtoms,
 T edgeLengthSlope,
 T edgeLengthOffset,
 int refinementLevel,
 geom::IndSimpSetIncAdj<3, 2, T>* mesh,
 IntOutputIterator actuallyClip,
 T maximumStretchFactor = 0,
 bool areUsingCircularEdges = false);


} // namespace mst
}

#define __mst_triangulateAtomRubber_ipp__
#include "stlib/mst/triangulateAtomRubber.ipp"
#undef __mst_triangulateAtomRubber_ipp__

#endif
