// -*- C++ -*-

#if !defined(__geom_mesh_iss_solveLaplacian_ipp__)
#error This file is an implementation detail.
#endif

namespace stlib
{
namespace geom {


template<std::size_t N, std::size_t M, typename T>
inline
void
solveLaplacian(IndSimpSetIncAdj<N, M, T>* mesh) {
   //
   // Determine which vertices are interior.
   //
   std::vector<bool> interior(mesh->vertices.size());
   // Interior indices.
   std::vector<int> ii(mesh->vertices.size(), -1);
   std::size_t numInterior = 0;
   for (std::size_t v = 0; v != mesh->vertices.size(); ++v) {
      if (mesh->isVertexOnBoundary(v)) {
         interior[v] = false;
      }
      else {
         interior[v] = true;
         ii[v] = numInterior++;
      }
   }

   TNT::Array2D<T> m(numInterior, numInterior, 0.0);
   TNT::Array1D<T> b(numInterior, 0.0);

   std::set<std::size_t> neighbors;

   std::size_t i, j;
   // For each space dimension.
   for (std::size_t n = 0; n != N; ++n) {

      //
      // Make the matrix, m, and the right hand side, b.
      //

      m = 0.0;
      b = 0.0;
      // For each interior vertex.
      for (std::size_t v = 0; v != mesh->vertices.size(); ++v) {
         if (interior[v]) {
            i = ii[v];
            // Get the neighboring vertex indices.
            getNeighbors(*mesh, v, neighbors);
            // The number of neighbors.
            m[i][i] = neighbors.size();
            // For each neighbor.
            for (std::set<std::size_t>::const_iterator iter = neighbors.begin();
                  iter != neighbors.end(); ++iter) {
               j = ii[*iter];
               // If the neighbor is an interior vertex.
               if (interior[*iter]) {
                  m[i][j] = - 1.0;
               }
               else {
                  b[i] += mesh->vertices[*iter][n];
               }
            }
         }
      }

      //
      // Solve the linear system.
      //

      JAMA::LU<T> lu(m);
      TNT::Array1D<T> x = lu.solve(b);
      // Set the vertex positions.
      for (std::size_t v = 0; v != mesh->vertices.size(); ++v) {
         if (interior[v]) {
            i = ii[v];
            mesh->vertices[v][n] = x[i];
         }
      }
   }
}

} // namespace geom
}
