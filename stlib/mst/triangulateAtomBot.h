// -*- C++ -*-

/*!
  \file triangulateAtomBot.h
  \brief Function for tesselating an atom using cut clipping with a bucket of triangles.
*/

#if !defined(__mst_triangulateAtomBot_h__)
#define __mst_triangulateAtomBot_h__

#include "stlib/mst/triangulateAtom.h"

namespace stlib
{
namespace mst
{


//! Triangulate the visible surface with a bucket of triangles.
/*!
  \param atom The atom to triangulate.
  \param clippingIdentifiers The identifiers of the atoms that might
  clip the surface.
  \param clippingAtoms The atoms that might clip the surface.
  \param edgeLengthSlope The slope of the maximum edge length function.
  \param edgeLengthOffset The offset of the maximum edge length function.
  \param refinementLevel The refinement level for the initial mesh.
  (Used only if it is non-negative.)
  \param outputTriangles The output triangles.
  \param actuallyClip The identifiers of the atoms that actually clip the mesh.

  \return The target edge length.
*/
template < typename T, typename TriangleOutputIterator,
           typename IntOutputIterator >
T
triangulateVisibleSurfaceWithBot(const geom::Ball<T, 3>& atom,
                                 std::vector<std::size_t>& clippingIdentifiers,
                                 std::vector< geom::Ball<T, 3> >& clippingAtoms,
                                 T edgeLengthSlope,
                                 T edgeLengthOffset,
                                 int refinementLevel,
                                 TriangleOutputIterator outputTriangles,
                                 IntOutputIterator actuallyClip);


} // namespace mst
}

#define __mst_triangulateAtomBot_ipp__
#include "stlib/mst/triangulateAtomBot.ipp"
#undef __mst_triangulateAtomBot_ipp__

#endif
