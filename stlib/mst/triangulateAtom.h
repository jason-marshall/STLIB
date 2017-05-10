// -*- C++ -*-

/*!
  \file triangulateAtom.h
  \brief Function for tesselating a unit sphere.
*/

#if !defined(__mst_triangulateAtom_h__)
#define __mst_triangulateAtom_h__

#include "stlib/mst/triangulateAtomRubber.h"
#include "stlib/mst/triangulateAtomCut.h"

#include "stlib/geom/mesh/iss/IndSimpSetIncAdj.h"

namespace stlib
{
namespace mst
{

//! Triangulate the visible surface using hybrid rubber/cut clipping.
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
  \param epsilon In cut clipping, if a vertex is within epsilon of the
  clipping surface it is considered to be on the clipping surface.
  \param areUsingCircularEdges specifies whether the edges should be
  interpreted as arcs of a great circle on the atomic surface.

  \return The target edge length.
*/
template<typename T, typename IntOutputIterator>
T
triangulateVisibleSurface(const geom::Ball<T, 3>& atom,
                          std::vector<std::size_t>& clippingIdentifiers,
                          std::vector<geom::Ball<T, 3> >& clippingAtoms,
                          T edgeLengthSlope,
                          T edgeLengthOffset,
                          int refinementLevel,
                          geom::IndSimpSetIncAdj<3, 2, T>* mesh,
                          IntOutputIterator actuallyClip,
                          T maximumStretchFactor = 0,
                          T epsilon = 0,
                          bool areUsingCircularEdges = false);


} // namespace mst
}

#define __mst_triangulateAtom_ipp__
#include "stlib/mst/triangulateAtom.ipp"
#undef __mst_triangulateAtom_ipp__

#endif
