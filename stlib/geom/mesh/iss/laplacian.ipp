// -*- C++ -*-

#if !defined(__geom_mesh_iss_laplacian_ipp__)
#error This file is an implementation detail.
#endif

namespace stlib
{
namespace geom {


// CONTINUE: Move
// Get the neighboring vertices of a vertex.
template<std::size_t N, std::size_t M, typename T>
inline
void
getNeighbors(const IndSimpSetIncAdj<N, M, T>& mesh,
             const std::size_t index,
             std::set<std::size_t>& indexSet) {
   indexSet.clear();

   std::size_t simp, vert;
   // For each incident simplex.
   for (std::size_t n = 0; n != mesh.incident.size(index); ++n) {
      // The index of the simplex.
      simp = mesh.incident(index, n);
      // For each vertex of the simplex.
      for (std::size_t m = 0; m != M + 1; ++m) {
         // The vertex index.
         vert = mesh.indexedSimplices[simp][m];
         // Don't include the index_th vertex.
         if (vert != index) {
            // Add it to the set (if it is not already in the set).
            indexSet.insert(vert);
         }
      }
   }
}


// CONTINUE This will give me boundary vertices that do not share a boundary
// face with the specified vertex.  Is this what I want?

// CONTINUE: Move
// Get the neighboring boundary vertices of a vertex.
template<std::size_t N, std::size_t M, typename T>
inline
void
getBoundaryNeighbors(const IndSimpSetIncAdj<N, M, T>& mesh,
                     const std::size_t index, std::set<std::size_t>& indexSet) {
   indexSet.clear();

   std::size_t simp, vert;
   // For each incident simplex.
   for (std::size_t n = 0; n != mesh.getIncidentSize(index); ++n) {
      // The index of the simplex.
      simp = mesh.getIncident(index, n);
      // For each vertex of the simplex.
      for (std::size_t m = 0; m != M + 1; ++m) {
         // The vertex index.
         vert = mesh.indexedSimplices[simp][m];
         // Don't include the index_th vertex.
         if (vert != index && mesh.isVertexOnBoundary(vert)) {
            // Add it to the set (if it is not already in the set).
            indexSet.insert(vert);
         }
      }
   }
}






template<std::size_t N, std::size_t M, typename T>
inline
void
applyLaplacian(IndSimpSetIncAdj<N, M, T>* mesh, const std::size_t numSweeps) {
   typedef IndSimpSetIncAdj<N, M, T> ISS;
   typedef typename ISS::Vertex Vertex;
   typedef typename ISS::Number Number;

   //
   // Determine the neighbors for the interior vertices.
   //
   container::PackedArrayOfArrays<std::size_t> neighbors;
   std::set<std::size_t> neighborsSet;
   for (std::size_t v = 0; v != mesh->vertices.size(); ++v) {
      if (mesh->isVertexOnBoundary(v)) {
         // Don't record neigbors for boundary vertices.
         neighbors.pushArray();
      }
      else {
         getNeighbors(*mesh, v, neighborsSet);
         assert(neighborsSet.size() != 0);
         neighbors.pushArray(neighborsSet.begin(), neighborsSet.end());
      }
   }

   Vertex pt;
   // Loop for the number of sweeps.
   for (std::size_t sweep = 0; sweep != numSweeps; ++sweep) {
      // For each vertex.
      for (std::size_t v = 0; v != mesh->vertices.size(); ++v) {
         // Skip boundary vertices.
         if (neighbors.empty(v)) {
            continue;
         }
         pt.fill(0);
         // For each neighbor.
         for (std::size_t i = 0; i != neighbors.size(v); ++i) {
            pt += mesh->vertices[neighbors(v, i)];
         }
         pt /= Number(neighbors.size(v));
         mesh->vertices[v] = pt;
      }
   }
}



// Perform sweeps of Laplacian smoothing on the boundary vertices.
template<typename T, class BoundaryCondition>
inline
void
applyLaplacian(IndSimpSetIncAdj<2, 1, T>* mesh,
               const BoundaryCondition& condition,
               const T maxAngleDeviation, std::size_t numSweeps) {
   typedef IndSimpSetIncAdj<2, 1, T> ISS;
   typedef typename ISS::Vertex Vertex;

   // The maximum cosine for moving a vertex.
   const T maxCosine = std::cos(numerical::Constants<T>::Pi() -
                                maxAngleDeviation);

   Vertex a;

   const std::size_t size = mesh->vertices.size();

   // Loop for the number of sweeps.
   while (numSweeps-- != 0) {
      // Loop over the vertices.
      for (std::size_t n = 0; n != size; ++n) {
         // Don't alter boundary vertices, they have only one incident edge.
         if (mesh->isVertexOnBoundary(n)) {
            continue;
         }

         // Don't alter the vertex if the angle is too sharp.
         if (computeCosineAngle(*mesh, n) > maxCosine) {
            continue;
         }

         //
         // Laplacian smoothing on the vertex.
         //

         // Averaging.
         a = getPreviousVertex(*mesh, n);
         a += getNextVertex(*mesh, n);
         a *= 0.5;
         // Set the position.
         mesh->vertices[n] = a;
         // Apply the boundary condition.
         applyBoundaryCondition(mesh, condition, n);
      }
   }
}




// Perform sweeps of Laplacian smoothing on the vertices.
template<typename T, std::size_t SD>
inline
void
applyLaplacian(IndSimpSetIncAdj<2, 1, T>* mesh,
               PointsOnManifold<2, 1, SD, T>* manifold,
               std::size_t numSweeps) {
   typedef IndSimpSetIncAdj<2, 1, T> ISS;
   typedef typename ISS::Vertex Vertex;

   Vertex x;

   const std::size_t size = mesh->vertices.size();

   // Loop for the number of sweeps.
   while (numSweeps-- != 0) {
      // Loop over the vertices.
      for (std::size_t n = 0; n != size; ++n) {
         // Don't alter corner vertices.
         if (manifold->isOnCorner(n)) {
            continue;
         }

         //
         // Laplacian smoothing on the vertex.
         //

         // Averaging.
         x = getPreviousVertex(*mesh, n);
         x += getNextVertex(*mesh, n);
         x *= 0.5;
         // Compute the closest point on the manifold.
         x = manifold->computeClosestPoint(n, x);
         // Update the point information on the manifold.
         manifold->updatePoint();
         // Set the position.
         mesh->vertices[n] = x;
      }
   }
}



// Perform sweeps of Laplacian smoothing on the boundary vertices.
template<typename T, class BoundaryCondition>
inline
void
applyLaplacian(IndSimpSetIncAdj<3, 2, T>* mesh,
               const BoundaryCondition& condition,
               const T maxAngleDeviation, std::size_t numSweeps) {
   typedef IndSimpSetIncAdj<3, 2, T> ISS;
   typedef typename ISS::Vertex Vertex;

   const std::size_t size = mesh->vertices.size();

   const bool areCheckingAngle =
      (maxAngleDeviation < 2 * numerical::Constants<T>::Pi() ? true : false);

   std::set<std::size_t> neighbors;
   Vertex pt;

   // Loop for the number of sweeps.
   while (numSweeps-- != 0) {
      // Loop over the vertices.
      for (std::size_t n = 0; n != size; ++n) {
         // Don't alter boundary vertices.
         if (mesh->isVertexOnBoundary(n)) {
            continue;
         }

         // The vertex should have at least three incident faces.
         assert(mesh->incident.size(n) >= 3);


         if (areCheckingAngle) {
            // Don't alter the vertex if the angle is too sharp.
            if (std::abs(computeAngle(*mesh, n) -
                         2 * numerical::Constants<T>::Pi()) > maxAngleDeviation) {
               continue;
            }
         }

         //
         // Laplacian smoothing on the vertex.
         //

         // Get the neighboring boundary vertex indices.
         getNeighbors(*mesh, n, neighbors);
         assert(neighbors.size() >= 3);
         std::fill(pt.begin(), pt.end(), 0.0);
         // For each neighbor.
         for (std::set<std::size_t>::const_iterator iter = neighbors.begin();
               iter != neighbors.end(); ++iter) {
            pt += mesh->vertices[*iter];
         }
         pt /= T(neighbors.size());
         mesh->vertices[n] = pt;
         // Apply the boundary condition.
         applyBoundaryCondition(mesh, condition, n);
      }
   }
}

} // namespace geom
}
