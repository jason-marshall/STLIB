// -*- C++ -*-

#if !defined(__geom_mesh_iss_distance_ipp__)
#error This file is an implementation detail.
#endif

namespace stlib
{
namespace geom {

//---------------------------------------------------------------------------
// Signed distance, 2-D space, 1-manifold, single point.
//---------------------------------------------------------------------------

// Compute the signed distance to the mesh and closest point on the mesh.
template<typename _T>
typename IndSimpSetIncAdj<2, 1, _T>::Number
computeSignedDistance
(const IndSimpSetIncAdj<2, 1, _T>& mesh,
 const std::vector<typename IndSimpSetIncAdj<2, 1, _T>::Number>&
 squaredHalfLengths,
 const typename IndSimpSetIncAdj<2, 1, _T>::Vertex& point,
 typename IndSimpSetIncAdj<2, 1, _T>::Vertex* closestPoint) {
   // Types.
   typedef typename IndSimpSetIncAdj<2, 1, _T>::Number Number;
   typedef typename IndSimpSetIncAdj<2, 1, _T>::Vertex Vertex;
   typedef typename IndSimpSetIncAdj<2, 1, _T>::Simplex Simplex;

   // Check for an empty mesh.
   if (mesh.indexedSimplices.empty()) {
      return std::numeric_limits<Number>::max();
   }
   assert(! mesh.vertices.empty());

   //
   // Find the closest vertex.
   //
   std::size_t closestVertexIndex = std::numeric_limits<std::size_t>::max();
   std::vector<Number> vertexSquaredDistances(mesh.vertices.size());
   Number minimumSquaredDistance = std::numeric_limits<Number>::max();
   for (std::size_t i = 0; i != mesh.vertices.size(); ++i) {
      vertexSquaredDistances[i] =
        ext::squaredDistance(point, mesh.vertices[i]);
      if (vertexSquaredDistances[i] < minimumSquaredDistance) {
         closestVertexIndex = i;
         minimumSquaredDistance = vertexSquaredDistances[i];
      }
   }
   assert(closestVertexIndex != std::size_t(-1));

   // Compute the signed distance to the closest vertex.
   *closestPoint = mesh.vertices[closestVertexIndex];
   Number signedDistance =
      computeSignedDistance(*closestPoint,
                            computeVertexNormal(mesh, closestVertexIndex), point);

   //
   // Check the faces.
   //
   Simplex face;
   Number sd;
   Vertex cp;
   for (std::size_t i = 0; i != mesh.indexedSimplices.size(); ++i) {
      const std::size_t sourceIndex = mesh.indexedSimplices[i][0];
      const std::size_t targetIndex = mesh.indexedSimplices[i][1];
      // A lower bound on the squared distance to the face.  Subtract the
      // square of half of the edge length from the squared distance to the
      // closer end point.
      Number lowerBound = std::min(vertexSquaredDistances[sourceIndex],
                                   vertexSquaredDistances[targetIndex]) -
                          squaredHalfLengths[i];
      if (lowerBound < minimumSquaredDistance) {
         // Compute the signed distance and closest point on the face.
         mesh.getSimplex(i, &face);
         sd = computeSignedDistance(face, point, &cp);
         // If the distance is smaller than previous ones.
         if (std::abs(sd) < std::abs(signedDistance)) {
            signedDistance = sd;
            *closestPoint = cp;
            minimumSquaredDistance = sd * sd;
         }
      }
   }

   return signedDistance;
}

// Compute the signed distance to the mesh and closest point on the mesh.
template<typename _T>
inline
typename IndSimpSetIncAdj<2, 1, _T>::Number
computeSignedDistance
(const IndSimpSetIncAdj<2, 1, _T>& mesh,
 const typename IndSimpSetIncAdj<2, 1, _T>::Vertex& point,
 typename IndSimpSetIncAdj<2, 1, _T>::Vertex* closestPoint) {
   typedef typename IndSimpSetIncAdj<2, 1, _T>::Number Number;
   std::vector<Number> squaredHalfLengths(mesh.indexedSimplices.size());
   for (std::size_t i = 0; i != mesh.indexedSimplices.size(); ++i) {
      squaredHalfLengths[i] = 0.25 *
        ext::squaredDistance(mesh.getSimplexVertex(i, 0),
                             mesh.getSimplexVertex(i, 1));
   }
   return computeSignedDistance(mesh, squaredHalfLengths, point, closestPoint);
}

//---------------------------------------------------------------------------
// Signed distance, 2-D space, 1-manifold, multiple points.
//---------------------------------------------------------------------------

//! Compute the signed distances to the mesh and closest points on the mesh.
template < typename _T,
         typename InputIterator, typename NumberOutputIterator,
         typename PointOutputIterator >
inline
void
computeSignedDistance
(const IndSimpSetIncAdj<2, 1, _T>& mesh,
 InputIterator pointsBeginning, InputIterator pointsEnd,
 NumberOutputIterator distances, PointOutputIterator closestPoints) {
   typedef typename IndSimpSetIncAdj<2, 1, _T>::Number Number;
   typedef typename IndSimpSetIncAdj<2, 1, _T>::Vertex Vertex;

   // Compute the square of the half edge lengths.
   std::vector<Number> squaredHalfLengths(mesh.indexedSimplices.size());
   for (std::size_t i = 0; i != mesh.indexedSimplices.size(); ++i) {
      squaredHalfLengths[i] = 0.25 *
        ext::squaredDistance(mesh.getSimplexVertex(i, 0),
                             mesh.getSimplexVertex(i, 1));
   }

   Vertex cp;
   for (; pointsBeginning != pointsEnd; ++pointsBeginning) {
      *distances++ = computeSignedDistance(mesh, squaredHalfLengths,
                                           *pointsBeginning, &cp);
      *closestPoints++ = cp;
   }
}

//---------------------------------------------------------------------------
// Signed distance, 3-D space, 2-manifold, single point.
//---------------------------------------------------------------------------

// Compute the signed distance to the mesh and closest point on the mesh.
template<typename _T>
typename IndSimpSetIncAdj<3, 2, _T>::Number
computeSignedDistance
(const IndSimpSetIncAdj<3, 2, _T>& mesh,
 const std::vector<typename IndSimpSetIncAdj<3, 2, _T>::Number>&
 squaredLongestEdgeLengths,
 const typename IndSimpSetIncAdj<3, 2, _T>::Vertex& point,
 typename IndSimpSetIncAdj<3, 2, _T>::Vertex* closestPoint) {
   // Types.
   typedef IndSimpSetIncAdj<3, 2, _T> Mesh;
   typedef typename Mesh::Number Number;
   typedef typename Mesh::Vertex Vertex;
   typedef typename Mesh::Simplex Simplex;

   const std::size_t M = Mesh::M;

   // Check for an empty mesh.
   if (mesh.indexedSimplices.empty()) {
      return std::numeric_limits<Number>::max();
   }
   assert(! mesh.vertices.empty());

   //
   // Find the closest vertex.
   //
   std::size_t closestVertexIndex = std::numeric_limits<std::size_t>::max();
   std::vector<Number> vertexSquaredDistances(mesh.vertices.size());
   Number minimumSquaredDistance = std::numeric_limits<Number>::max();
   for (std::size_t i = 0; i != mesh.vertices.size(); ++i) {
      vertexSquaredDistances[i] =
        ext::squaredDistance(point, mesh.vertices[i]);
      if (vertexSquaredDistances[i] < minimumSquaredDistance) {
         closestVertexIndex = i;
         minimumSquaredDistance = vertexSquaredDistances[i];
      }
   }
   assert(closestVertexIndex != std::size_t(-1));

   // Compute the signed distance to the closest vertex.
   *closestPoint = mesh.vertices[closestVertexIndex];
   Number signedDistance =
      computeSignedDistance(*closestPoint,
                            computeVertexNormal(mesh, closestVertexIndex), point);

   //
   // Check the faces and edges.
   //
   Simplex face;
   std::array < Vertex, 1 + 1 > edge;
   Number sd;
   Vertex cp, faceNormal, edgeNormal;
   for (std::size_t i = 0; i != mesh.indexedSimplices.size(); ++i) {
      // A lower bound on the squared distance to the face.  Subtract the
      // square of the longest edge length (of the simplex) from the squared
      // distance to the closest vertex (of the simplex).
      Number lowerBound =
         ads::min(vertexSquaredDistances[mesh.indexedSimplices[i][0]],
                  vertexSquaredDistances[mesh.indexedSimplices[i][1]],
                  vertexSquaredDistances[mesh.indexedSimplices[i][2]]) -
         squaredLongestEdgeLengths[i];
      if (lowerBound < minimumSquaredDistance) {
         // Compute the signed distance and closest point on the face.
         mesh.getSimplex(i, &face);
         computeSimplexNormal(mesh, i, &faceNormal);
         sd = computeSignedDistance(face, faceNormal, point, &cp);
         // If the distance is smaller than previous ones.
         if (std::abs(sd) < std::abs(signedDistance)) {
            signedDistance = sd;
            *closestPoint = cp;
            minimumSquaredDistance = sd * sd;
         }
         // Compute the signed distance and closest point on the on the
         // appropriate edges.  (We don't need to check edges that will be
         // processed by other faces.)
         for (std::size_t n = 0; n != M + 1; ++n) {
            std::size_t adjacentIndex = mesh.adjacent[i][n];
            // If this face is responsible for this edge.
            if (i < adjacentIndex) {
               // Make the edge.
               edge[0] = face[(n + 1) % (M + 1)];
               edge[1] = face[(n + 2) % (M + 1)];
               // Calculate the edge normal.
               computeSimplexNormal(mesh, i, &edgeNormal);
               computeSimplexNormal(mesh, adjacentIndex, &faceNormal);
               edgeNormal += faceNormal;
               ext::normalize(&edgeNormal);
               // Compute the signed distance and closest point on the edge.
               sd = computeSignedDistance(edge, edgeNormal, point, &cp);
               // If the distance is smaller than previous ones.
               if (std::abs(sd) < std::abs(signedDistance)) {
                  signedDistance = sd;
                  *closestPoint = cp;
                  minimumSquaredDistance = sd * sd;
               }
            }
         }
      }
   }

   return signedDistance;
}

// Compute the signed distance to the mesh and closest point on the mesh.
template<typename _T>
inline
typename IndSimpSetIncAdj<3, 2, _T>::Number
computeSignedDistance
(const IndSimpSetIncAdj<3, 2, _T>& mesh,
 const typename IndSimpSetIncAdj<3, 2, _T>::Vertex& point,
 typename IndSimpSetIncAdj<3, 2, _T>::Vertex* closestPoint) {
   typedef typename IndSimpSetIncAdj<3, 2, _T>::Number Number;

   // Compute the square of the longest edge lengths.
   std::vector<Number> squaredLongestEdgeLengths(mesh.indexedSimplices.size());
   for (std::size_t i = 0; i != mesh.indexedSimplices.size(); ++i) {
      squaredLongestEdgeLengths[i] =
         ads::max(ext::squaredDistance(mesh.getSimplexVertex(i, 0),
                                       mesh.getSimplexVertex(i, 1)),
                  ext::squaredDistance(mesh.getSimplexVertex(i, 1),
                                       mesh.getSimplexVertex(i, 2)),
                  ext::squaredDistance(mesh.getSimplexVertex(i, 2),
                                       mesh.getSimplexVertex(i, 0)));
   }

   return computeSignedDistance(mesh, squaredLongestEdgeLengths, point,
                                closestPoint);
}

//---------------------------------------------------------------------------
// Signed distance, 3-D space, 2-manifold, multiple points.
//---------------------------------------------------------------------------

//! Compute the signed distances to the mesh and closest points on the mesh.
template < typename _T,
         typename InputIterator, typename NumberOutputIterator,
         typename PointOutputIterator >
inline
void
computeSignedDistance
(const IndSimpSetIncAdj<3, 2, _T>& mesh,
 InputIterator pointsBeginning, InputIterator pointsEnd,
 NumberOutputIterator distances, PointOutputIterator closestPoints) {
   typedef typename IndSimpSetIncAdj<3, 2, _T>::Number Number;
   typedef typename IndSimpSetIncAdj<3, 2, _T>::Vertex Vertex;

   // Compute the square of the longest edge lengths.
   std::vector<Number> squaredLongestEdgeLengths(mesh.indexedSimplices.size());
   for (std::size_t i = 0; i != mesh.indexedSimplices.size(); ++i) {
      squaredLongestEdgeLengths[i] =
         ads::max(ext::squaredDistance(mesh.getSimplexVertex(i, 0),
                                       mesh.getSimplexVertex(i, 1)),
                  ext::squaredDistance(mesh.getSimplexVertex(i, 1),
                                       mesh.getSimplexVertex(i, 2)),
                  ext::squaredDistance(mesh.getSimplexVertex(i, 2),
                                       mesh.getSimplexVertex(i, 0)));
   }

   Vertex cp;
   for (; pointsBeginning != pointsEnd; ++pointsBeginning) {
      *distances++ = computeSignedDistance(mesh, squaredLongestEdgeLengths,
                                           *pointsBeginning, &cp);
      *closestPoints++ = cp;
   }
}

} // namespace geom
}
