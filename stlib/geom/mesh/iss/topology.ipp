// -*- C++ -*-

#if !defined(__geom_mesh_iss_topology_ipp__)
#error This file is an implementation detail.
#endif

namespace stlib
{
namespace geom {

// Count the connected components of the mesh.
template<std::size_t N, std::size_t M, typename T>
inline
std::size_t
countComponents(const IndSimpSetIncAdj<N, M, T>& mesh) {
   // Which simplices have been identified as belonging to a particular
   // component.
   std::vector<bool> used(mesh.indexedSimplices.size(), false);

   // Simplex indices in a single component.
   std::vector<std::size_t> indices;
   std::size_t i, iEnd;
   std::size_t n = 0;
   // The number of components.
   std::size_t numComponents = 0;
   // The number of simplices accounted for so far.
   std::size_t numSimplices = 0;
   do {
      ++numComponents;
      // Get the first unused simplex.
      while (used[n]) {
         ++n;
      }
      // Get the component that contains the n_th simplex.
      determineSimplicesInComponent(mesh, n, std::back_inserter(indices));

      // Record the simplices that are used in this component.
      iEnd = indices.size();
      for (i = 0; i != iEnd; ++i) {
         used[indices[i]] = true;
      }

      numSimplices += indices.size();
      indices.clear();
   }
   while (numSimplices != mesh.indexedSimplices.size());

   return numComponents;
}

// Return true if the simplices share a vertex.
template<std::size_t N, std::size_t M, typename T>
bool
doSimplicesShareAnyVertex(const IndSimpSetIncAdj<N, M, T>& mesh,
                          const std::size_t i, const std::size_t j) {
   // For each vertex of the first simplex.
   for (std::size_t m = 0; m != M + 1; ++m) {
      // The m_th vertex if the first simplex.
      const std::size_t v = mesh.indexedSimplices[i][m];
      // For each simplex incident to the vertex.
      for (std::size_t s = 0; s != mesh.incident.size(v); ++s) {
         // If the second simplex is incident to the vertex.
         if (mesh.incident(v, s) == j) {
            return true;
         }
      }
   }
   return false;
}

} // namespace geom
}
