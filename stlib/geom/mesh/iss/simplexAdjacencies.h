// -*- C++ -*-

/*!
  \file geom/mesh/iss/simplexAdjacencies.h
  \brief Simplex-simplex adjacencies in an indexed simplex set.
*/

#if !defined(__geom_simplexAdjacencies_h__)
#define __geom_simplexAdjacencies_h__

#include "stlib/geom/mesh/iss/vertexSimplexIncidence.h"

#include "stlib/geom/kernel/simplexTopology.h"
#include "stlib/container/StaticArrayOfArrays.h"

namespace stlib
{
namespace geom {

//! Simplex-simplex adjacencies in an indexed simplex set.
template<std::size_t SpaceD>
inline
void
simplexAdjacencies
(std::vector<std::array<std::size_t, SpaceD> >* adjacent,
 const std::vector<std::array<std::size_t, SpaceD> >& simplices,
 const container::StaticArrayOfArrays<std::size_t>& vertexSimplexInc) {
   // The container for the adjacent simplex indices.
   typedef std::array<std::size_t, SpaceD> IndexContainer;

   // Allocate memory for the adjacencies.
   adjacent->resize(simplices.size());


   // Initialize the adjacent indices to a null value.
   std::fill(adjacent->begin(), adjacent->end(),
             ext::filled_array<IndexContainer>
             (std::numeric_limits<std::size_t>::max()));

   std::size_t m, j, vertexIndex, simplexIndex, numIncident;
   const std::size_t sz = simplices.size();
   typename std::array<std::size_t, SpaceD-1> face;
   // For each simplex.
   for (std::size_t i = 0; i != sz; ++i) {
      // For each vertex of the simplex
      for (m = 0; m != SpaceD; ++m) {
         // Get the face opposite the vertex.
         getFace(simplices[i], m, &face);
         // A vertex on the face.
         vertexIndex = face[0];
         // For each simplex that is incident to the vertex.
         numIncident = vertexSimplexInc.size(vertexIndex);
         for (j = 0; j != numIncident; ++j) {
            simplexIndex = vertexSimplexInc(vertexIndex, j);
            if (i != simplexIndex && hasFace(simplices[simplexIndex], face)) {
               (*adjacent)[i][m] = simplexIndex;
               break;
            }
         }
      }
   }
}

//! Simplex-simplex adjacencies in an indexed simplex set.
template<std::size_t SpaceD>
inline
void
simplexAdjacencies
(std::vector<std::array<std::size_t, SpaceD> >* adjacent,
 const std::size_t numVertices,
 const std::vector<std::array<std::size_t, SpaceD> >& simplices) {
   // Build the incidence relationships.
   container::StaticArrayOfArrays<std::size_t> vertexSimplexInc;
   vertexSimplexIncidence(&vertexSimplexInc, numVertices, simplices);
   // Compute the adjacencies.
   simplexAdjacencies(adjacent, simplices, vertexSimplexInc);
}

//! Return number of simplices adjacent to the given simplex.
/*!
  \c std::numeric_limits<std::size_t>::max() represents an invalid index
  (no adjacent simplex).
*/
template<std::size_t SpaceD>
inline
std::size_t
numAdjacent(const std::array<std::size_t, SpaceD>& adjacencies) {
   return SpaceD - std::count(adjacencies.begin(), adjacencies.end(),
                              std::numeric_limits<std::size_t>::max());
}

} // namespace geom
}

#endif
