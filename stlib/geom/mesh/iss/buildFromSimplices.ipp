// -*- C++ -*-

#if !defined(__geom_mesh_iss_buildFromSimplices_ipp__)
#error This file is an implementation detail.
#endif

namespace stlib
{
namespace geom {


template < std::size_t N, std::size_t M, typename T,
         typename VertexForIter >
inline
void
buildFromSimplices(VertexForIter verticesBeginning,
                   VertexForIter verticesEnd,
                   IndSimpSet<N, M, T>* mesh) {
   typedef IndSimpSet<N, M, T> ISS;
   typedef typename ISS::Vertex Vertex;

   // Used to hold the distinct vertices.
   std::vector<Vertex> distinct;

   // Resize the simplices array.
   const std::size_t numInputVertices =
      std::distance(verticesBeginning, verticesEnd);
   assert(numInputVertices % (M + 1) == 0);
   mesh->indexedSimplices.resize(numInputVertices / (M + 1));

   // Make the indexed set of distinct vertices.
   std::vector<std::size_t> indices;
   indices.reserve(numInputVertices);
   buildDistinctPoints<N>(verticesBeginning, verticesEnd,
                          std::back_inserter(distinct),
                          std::back_inserter(indices));

   // The indices are the indexed simplices.
   std::size_t n = 0;
   for (std::size_t i = 0; i != mesh->indexedSimplices.size(); ++i) {
      for (std::size_t m = 0; m != M + 1; ++m) {
         mesh->indexedSimplices[i][m] = indices[n++];
      }
   }

   // Copy the distinct vertices into the vertex array.
   mesh->vertices.resize(distinct.size());
   std::copy(distinct.begin(), distinct.end(), mesh->vertices.begin());

   // Update any auxilliary topological information.
   mesh->updateTopology();
}

} // namespace geom
}
