// -*- C++ -*-

#if !defined(__geom_mesh_iss_contact_ipp__)
#error This file is an implementation detail.
#endif

namespace stlib
{
namespace geom {


// Move the vertices to remove contact.
template < std::size_t N, typename T,
         typename VertexForwardIterator >
inline
std::size_t
removeContact(const IndSimpSet < N, N - 1, T > & surface,
              VertexForwardIterator verticesBeginning,
              VertexForwardIterator verticesEnd) {
   typedef IndSimpSet < N, N - 1, T > Mesh;
   typedef typename Mesh::Vertex Vertex;
   typedef ISS_SignedDistance<Mesh> SignedDistance;

   // Make the signed distance data structure.
   SignedDistance signedDistance(surface);

   std::size_t count = 0;
   Vertex closestPoint;
   // For each vertex.
   for (; verticesBeginning != verticesEnd; ++verticesBeginning) {
      // If the vertex is inside the object.
      if (signedDistance(*verticesBeginning, &closestPoint) < 0) {
         ++count;
         // Move the vertex to the closest point.
         *verticesBeginning = closestPoint;
      }
   }
   return count;
}

} // namespace geom
}
