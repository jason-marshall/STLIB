// -*- C++ -*-

#if !defined(__geom_mesh_iss_fit_ipp__)
#error This file is an implementation detail.
#endif

namespace stlib
{
namespace geom {


template<typename T, class ISS>
inline
void
fit(IndSimpSetIncAdj<2, 1, T>* mesh,
    const ISS_SignedDistance<ISS, 2>& signedDistance,
    const T deviationTangent) {
   typedef IndSimpSetIncAdj<2, 1, T> Mesh;
   typedef typename Mesh::Vertex Vertex;

   std::size_t i, j;
   std::size_t previousVertexIndex, nextVertexIndex;
   Vertex point;
   Vertex tangent;
   T previousDistance, nextDistance, previousDeviation, nextDeviation;

   // Loop over the vertices.
   const std::size_t size = mesh->vertices.size();
   for (std::size_t n = 0; n != size; ++n) {
      // Don't alter boundary vertices.
      if (mesh->isVertexOnBoundary(n)) {
         continue;
      }

      //
      // Determine the deviation tangents.
      //

      // The vertex should have two incident faces.
      assert(mesh->incident.size(n) == 2);
      // The two incident edges.
      i = mesh->incident(n, 0);
      j = mesh->incident(n, 1);
      if (mesh->indexedSimplices[i][0] == n) {
         std::swap(i, j);
      }
      assert(mesh->indexedSimplices[i][1] == n &&
             mesh->indexedSimplices[j][0] == n);
      // Now i is the previous edge and j is the next edge.
      previousVertexIndex = mesh->indexedSimplices[i][0];
      nextVertexIndex = mesh->indexedSimplices[j][1];

      // The distance at the mid-points of the edges.
      point = mesh->vertices[n];
      point += mesh->vertices[previousVertexIndex];
      point *= 0.5;
      previousDistance = signedDistance(point);

      point = mesh->vertices[n];
      point += mesh->vertices[nextVertexIndex];
      point *= 0.5;
      nextDistance = signedDistance(point);

      /* REMOVE
      std::cout << "previousDistance = " << previousDistance
            << ", nextDistance = " << nextDistance << "\n";
      */

      // If the deviation is small, don't do anything with this vertex.
      previousDeviation = 2.0 * std::abs(previousDistance) /
        ext::euclideanDistance(mesh->vertices[previousVertexIndex],
                               mesh->vertices[n]);
      nextDeviation = 2.0 * std::abs(nextDistance) /
        ext::euclideanDistance(mesh->vertices[nextVertexIndex],
                               mesh->vertices[n]);

      if (previousDeviation < deviationTangent &&
            nextDeviation < deviationTangent) {
         continue;
      }

      // The offset distance.
#if 0
      if (previousDistance > 0 && nextDistance > 0) {
         //offset = - 2.0 * std::abs(previousDistance - nextDistance);
         offset = - 2.0 * std::max(std::abs(previousDistance),
                                   std::abs(nextDistance));
      }
      else if (previousDistance < 0 && nextDistance < 0) {
         //offset = 2.0 * std::abs(previousDistance - nextDistance);
         offset = 2.0 * std::max(std::abs(previousDistance),
                                 std::abs(nextDistance));
      }
      else {
         offset = - 2.0 * (previousDistance + nextDistance);
      }
      offset = - 2.0 * (previousDistance + nextDistance);
#endif

#if 0
      //
      // Compute the outward normal direction.
      //
      normal = 0.0;

      point = mesh->vertices[previousVertexIndex];
      point -= mesh->vertices[n];
      rotatePiOver2(&point);
      normalize(&point);
      normal += point;

      point = mesh->vertices[nextVertexIndex];
      point -= mesh->vertices[n];
      rotateMinusPiOver2(&point);
      normalize(&point);
      normal += point;

      normalize(&normal);
#endif

      //
      // Find the tangent to the surface.
      //

      tangent = signedDistance.computeNormal(mesh->vertices[n]);
      rotatePiOver2(&tangent);
      // The offset.
      tangent *= 2.0 * (std::abs(nextDistance) - std::abs(previousDistance));

      //
      // Offset the vertex and find the closest point.
      //
      point = mesh->vertices[n];
      point += tangent;
      mesh->vertices[n] = signedDistance.computeClosestPoint(point);
      /* REMOVE
      std::cout << tangent << " "
            << mesh->vertices()[n] << "\n";
      */
   }
}



template<typename T, class ISS>
inline
void
fit(IndSimpSetIncAdj<2, 1, T>* mesh,
    const ISS_SignedDistance<ISS, 2>& signedDistance,
    const T deviationTangent, std::size_t numSweeps) {
   while (numSweeps-- > 0) {
      fit(mesh, signedDistance, deviationTangent);
   }
}



// Fit the boundary of a mesh to a level-set description.
template<typename T, class ISS>
inline
void
fit(IndSimpSetIncAdj<2, 2, T>* mesh,
    const ISS_SignedDistance<ISS, 2>& signedDistance,
    const T deviationTangent,
    std::size_t numSweeps) {
   typedef IndSimpSetIncAdj<2, 1, T> Boundary;

   Boundary boundary;
   std::vector<std::size_t> boundaryVertexIndices;

   // Get the boundary.
   buildBoundary(*mesh, &boundary,
                 std::back_inserter(boundaryVertexIndices));

   // Fit the boundary to the level-set.
   fit(&boundary, signedDistance, deviationTangent, numSweeps);

   // Update the vertices in the mesh->
   const std::size_t size = boundaryVertexIndices.size();
   assert(size == boundary.vertices.size());
   for (std::size_t n = 0; n != size; ++n) {
      mesh->vertices[boundaryVertexIndices[n]] = boundary.vertices[n];
   }
}

} // namespace geom
}
