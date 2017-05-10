// -*- C++ -*-

#if !defined(__geom_VertexSimplexInc_ipp__)
#error This file is an implementation detail of the class VertexSimplexInc.
#endif

namespace stlib
{
namespace geom {


//
// Manipulators
//


template<std::size_t M>
template<typename IS>
inline
void
VertexSimplexInc<M>::
build(const std::size_t numVertices, const std::vector<IS>& simplices) {
   // Clear the old incidence information.
   clear();

   //
   // Determine the number of simplices incident to each vertex.
   //

   std::vector<std::size_t> numIncident(numVertices, std::size_t(0));
   std::size_t m;
   // Loop over the simplices.
   for (std::size_t i = 0; i != simplices.size(); ++i) {
      // Loop over the vertices of this simplex.
      for (m = 0; m != M + 1; ++m) {
         ++numIncident[simplices[i][m]];
      }
   }

   //
   // Build the incidence array with the size information.
   //

   _inc.rebuild(numIncident.begin(), numIncident.end());

   //
   // Fill the incidence array with the simplex indices.
   //

   // Vertex-simplex incidences.
   std::vector<Iterator> vsi(numVertices);
   for (std::size_t i = 0; i != numVertices; ++i) {
      vsi[i] = _inc(i);
   }
   // vertex index.
   std::size_t vi;
   // Loop over the simplices.
   for (std::size_t i = 0; i != simplices.size(); ++i) {
      // Loop over the vertices of this simplex.
      for (m = 0; m != M + 1; ++m) {
         // Add the simplex to the vertex incidence array.
         vi = simplices[i][m];
         *vsi[vi] = i;
         ++vsi[vi];
      }
   }

#ifdef STLIB_DEBUG
   // Check that we added the correct number of incidences for each vertex.
   for (std::size_t i = 0; i != numVertices; ++i) {
      assert(vsi[i] == _inc.end(i));
   }
#endif
}


//
// File output.
//


template<std::size_t M>
std::ostream&
operator<<(std::ostream& out, const VertexSimplexInc<M>& x) {
   x.put(out);
   return out;
}

} // namespace geom
}
