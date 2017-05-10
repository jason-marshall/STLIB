// -*- C++ -*-

#if !defined(__geom_SimplexAdj_ipp__)
#error This file is an implementation detail of the class SimplexAdj.
#endif

namespace stlib
{
namespace geom {

//
// Manipulators
//

template<std::size_t M>
inline
void
SimplexAdj<M>::
build(const std::vector < std::array < std::size_t, M + 1 > > & simplices,
      const container::StaticArrayOfArrays<std::size_t>& vertexSimplexInc) {
   // Allocate memory for the adjacencies.
   _adj.resize(simplices.size());


   // Initialize the adjacent indices to a null value.
   std::fill(_adj.begin(), _adj.end(),
             ext::filled_array<IndexContainer>
             (std::numeric_limits<std::size_t>::max()));

   std::size_t m, j, vertexIndex, simplexIndex, numIncident;
   const std::size_t sz = simplices.size();
   typename std::array<std::size_t, M> face;
   // For each simplex.
   for (std::size_t i = 0; i != sz; ++i) {
      // For each vertex of the simplex
      for (m = 0; m != M + 1; ++m) {
         // Get the face opposite the vertex.
         getFace(simplices[i], m, &face);
         // A vertex on the face.
         vertexIndex = face[0];
         // For each simplex that is incident to the vertex.
         numIncident = vertexSimplexInc.size(vertexIndex);
         for (j = 0; j != numIncident; ++j) {
            simplexIndex = vertexSimplexInc(vertexIndex, j);
            if (i != simplexIndex && hasFace(simplices[simplexIndex], face)) {
               _adj[i][m] = simplexIndex;
               break;
            }
         }
      }
   }
}

} // namespace geom
}
