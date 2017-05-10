// -*- C++ -*-

#if !defined(__geom_mesh_iss_onManifold_ipp__)
#error This file is an implementation detail.
#endif

namespace stlib
{
namespace geom {

// Get the vertices on the manifold.
template < std::size_t N, std::size_t M, typename T,
         std::size_t MM, typename MT, typename IntOutIter >
inline
void
determineVerticesOnManifold(const IndSimpSet<N, M, T>& mesh,
                            const IndSimpSet<N, MM, MT>& manifold,
                            IntOutIter indexIterator,
                            const T epsilon) {
   // Call the function below with the full range of vertex indices.
   determineVerticesOnManifold
   (mesh, ads::constructIntIterator(std::size_t(0)),
    ads::constructIntIterator(mesh.vertices.size()),
    manifold, indexIterator, epsilon);
}



// Get the vertices (from the set of vertices) on the manifold.
template < std::size_t N, std::size_t M, typename T,
         typename IntInIter,
         std::size_t MM, typename MT, typename IntOutIter >
inline
void
determineVerticesOnManifold(const IndSimpSet<N, M, T>& mesh,
                            IntInIter indicesBeginning, IntInIter indicesEnd,
                            const IndSimpSet<N, MM, MT>& manifold,
                            IntOutIter indexIterator,
                            const T epsilon) {
   typedef IndSimpSet<N, MM, MT> Manifold;

   static_assert(MM < M, "The manifold dimension must be less than M.");

   // The data structure for computing unsigned distance.
   ISS_SimplexQuery<Manifold> distance(manifold);

   std::size_t i;
   for (; indicesBeginning != indicesEnd; ++indicesBeginning) {
      i = *indicesBeginning;
      if (distance.computeMinimumDistance(mesh.vertices[i]) <= epsilon) {
         *indexIterator++ = i;
      }
   }
}

} // namespace geom
}
