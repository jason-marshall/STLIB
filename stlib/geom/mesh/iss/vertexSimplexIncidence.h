// -*- C++ -*-

/*!
  \file geom/mesh/iss/vertexSimplexIncidence.h
  \brief Vertex-simplex incidences in an indexed simplex set.
*/

#if !defined(__geom_vertexSimplexIncidence_h__)
#define __geom_vertexSimplexIncidence_h__

#include "stlib/container/StaticArrayOfArrays.h"

namespace stlib
{
namespace geom {

//! Vertex-simplex incidences in an M-D indexed simplex set.
/*!
Construct the incidence relationships from the number of vertices and
the vector of indexed simplices. The simplices that are incident to each
vertex are stored in a packed array. \c IndexedSimplex is an tuple of M+1
integers, where M is the topological dimension of the simplices. It must be
subscriptable.
*/
template<typename IndexedSimplex>
inline
void
vertexSimplexIncidence(container::StaticArrayOfArrays<std::size_t>* incident,
                       const std::size_t numVertices,
                       const std::vector<IndexedSimplex>& simplices) {
   // An iterator on the vertex-simplex incidences.
   typedef container::StaticArrayOfArrays<std::size_t>::iterator Iterator;

   //
   // Determine the number of simplices incident to each vertex.
   //

   std::vector<std::size_t> numIncident(numVertices, std::size_t(0));
   // Loop over the simplices.
   for (std::size_t i = 0; i != simplices.size(); ++i) {
      // Loop over the vertices of this simplex.
      for (std::size_t m = 0; m != simplices[i].size(); ++m) {
         ++numIncident[simplices[i][m]];
      }
   }

   //
   // Build the incidence array with the size information.
   //

   incident->rebuild(numIncident.begin(), numIncident.end());

   //
   // Fill the incidence array with the simplex indices.
   //

   // Vertex-simplex incidences.
   std::vector<Iterator> vsi(numVertices);
   for (std::size_t i = 0; i != numVertices; ++i) {
      vsi[i] = (*incident)(i);
   }
   // vertex index.
   std::size_t vi;
   // Loop over the simplices.
   for (std::size_t i = 0; i != simplices.size(); ++i) {
      // Loop over the vertices of this simplex.
      for (std::size_t m = 0; m != simplices[i].size(); ++m) {
         // Add the simplex to the vertex incidence array.
         vi = simplices[i][m];
         *vsi[vi] = i;
         ++vsi[vi];
      }
   }

#ifdef STLIB_DEBUG
   // Check that we added the correct number of incidences for each vertex.
   for (std::size_t i = 0; i != numVertices; ++i) {
      assert(vsi[i] == incident->end(i));
   }
#endif

   // Sort each of the incident sequences. This makes it easier to 
   // apply set operations. For example, the set of simplices that are
   // incident to an edge is the intersection of the sets that are incident
   // to the two end vertices.
   for (std::size_t i = 0; i != incident->getNumberOfArrays(); ++i) {
      std::sort(incident->begin(i), incident->end(i));
   }
}

} // namespace geom
}

#endif
