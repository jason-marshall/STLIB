// -*- C++ -*-

#if !defined(__mst_triangulateAtom_ipp__)
#error This file is an implementation detail of triangulateAtom.
#endif

namespace stlib
{
namespace mst
{


template<typename T, typename IntOutputIterator>
inline
T
triangulateVisibleSurface(const geom::Ball<T, 3>& atom,
                          std::vector<std::size_t>& clippingIdentifiers,
                          std::vector<geom::Ball<T, 3> >& clippingAtoms,
                          const T edgeLengthSlope,
                          const T edgeLengthOffset,
                          const int refinementLevel,
                          geom::IndSimpSetIncAdj<3, 2, T>* mesh,
                          IntOutputIterator actuallyClip,
                          const T maximumStretchFactor,
                          const T epsilon,
                          const bool areUsingCircularEdges)
{
  // Compute the clipping atoms and form the initial tesselation.
  const T targetEdgeLength =
    computeClippingAtomsAndTesselate(atom, clippingIdentifiers,
                                     clippingAtoms, edgeLengthSlope,
                                     edgeLengthOffset, refinementLevel,
                                     mesh, actuallyClip);

  // If the atom is completely erased by the clipping.
  if (mesh->indexedSimplices.size() == 0) {
    return targetEdgeLength;
  }

  // If we allow any stretching.
  if (maximumStretchFactor != 1) {
    clipWithRubberClipping(atom, clippingIdentifiers, clippingAtoms, mesh,
                           actuallyClip, maximumStretchFactor,
                           areUsingCircularEdges);
  }
  clipWithCutClipping(atom, clippingIdentifiers, clippingAtoms,
                      mesh, actuallyClip, epsilon);

  return targetEdgeLength;
}


} // namespace mst
}
