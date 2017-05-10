// -*- C++ -*-

#if !defined(__geom_mesh_iss_tile_ipp__)
#error This file is an implementation detail.
#endif

namespace stlib
{
namespace geom {


template<typename T, class LSF>
inline
void
tile(const BBox<T, 2>& domain, const T length, const LSF& f,
     IndSimpSet<2, 2, T>* mesh) {
   typedef typename IndSimpSet<2, 2, T>::Vertex Vertex;
   typedef typename IndSimpSet<2, 2, T>::IndexedSimplex IndexedSimplex;

   assert(! isEmpty(domain) && length > 0);

   // The height of the equilateral triangle.
   const T height = std::sqrt(3.) * length / 2;

   // The number of triangles in the x direction.
   const std::size_t numX = std::size_t(std::ceil((domain.upper[0] -
                                        domain.lower[0] +
                                        length / 2.) / length));
   // The number of triangles in the y direction.
   const std::size_t numY = std::size_t(std::ceil((domain.upper[1] -
                                        domain.lower[1]) /
                                        height));

   // The number of vertices and simplices in the mesh.
   const std::size_t numVertices = (numX + 1) * (numY + 1);

   // Resize the arrays.
   mesh->vertices.resize(numVertices);

   //
   // Make the vertices.
   //

   std::size_t i, j;
   Vertex x;
   std::size_t n = 0;
   // For each row of vertices.
   for (j = 0; j <= numY; ++j) {
      // Calculate the first vertex.
      x = domain.lower;
      if (j % 2 == 1) {
         x[0] -= length / 2.0;
      }
      x[1] += j * height;
      // For each vertex in the row.
      for (i = 0; i <= numX; ++i, ++n, x[0] += length) {
         mesh->vertices[n] = x;
      }
   }

   //
   // Make the simplices.
   //

   std::size_t row, col;
   IndexedSimplex s;
   std::vector<IndexedSimplex> simp;
   std::size_t m;
   for (row = 0; row != numY; ++row) {
      i = row * (numX + 1);
      j = i + numX + 1;
      for (col = 0; col != numX; ++col, ++i, ++j) {
         if (row % 2 == 0) {
            // The indexed simplex.
            s[0] = i;
            s[1] = i + 1;
            s[2] = j + 1;
            // Compute the centroid.
            std::fill(x.begin(), x.end(), 0);
            for (m = 0; m != 3; ++m) {
               x += mesh->vertices[ s[m] ];
            }
            x /= 3.0;
            // If the centroid is inside.
            if (f(x) <= 0) {
               // Add the indexed simplex.
               simp.push_back(s);
            }

            // The indexed simplex.
            s[0] = i;
            s[1] = j + 1;
            s[2] = j;
            // Compute the centroid.
            std::fill(x.begin(), x.end(), 0);
            for (m = 0; m != 3; ++m) {
               x += mesh->vertices[ s[m] ];
            }
            x /= 3.0;
            // If the centroid is inside.
            if (f(x) <= 0) {
               // Add the indexed simplex.
               simp.push_back(s);
            }
         }
         else {
            // The indexed simplex.
            s[0] = i;
            s[1] = i + 1;
            s[2] = j;
            // Compute the centroid.
            std::fill(x.begin(), x.end(), 0);
            for (m = 0; m != 3; ++m) {
               x += mesh->vertices[ s[m] ];
            }
            x /= 3.0;
            // If the centroid is inside.
            if (f(x) <= 0) {
               // Add the indexed simplex.
               simp.push_back(s);
            }

            // The indexed simplex.
            s[0] = i + 1;
            s[1] = j + 1;
            s[2] = j;
            // Compute the centroid.
            std::fill(x.begin(), x.end(), 0);
            for (m = 0; m != 3; ++m) {
               x += mesh->vertices[ s[m] ];
            }
            x /= 3.0;
            // If the centroid is inside.
            if (f(x) <= 0) {
               // Add the indexed simplex.
               simp.push_back(s);
            }
         }
      }
   }
   // Add the simplices whose centroids are inside the object to the mesh.
   mesh->indexedSimplices.resize(simp.size());
   std::copy(simp.begin(), simp.end(), mesh->indexedSimplices.begin());
   // Pack the mesh to get rid of unused vertices.
   pack(mesh);
}



template<typename T, class LSF>
inline
void
tile(const BBox<T, 3>& domain, const T length, const LSF& f,
     IndSimpSet<3, 3, T>* mesh) {
   typedef typename IndSimpSet<3, 3, T>::Vertex Vertex;
   typedef typename IndSimpSet<3, 3, T>::Simplex Simplex;

   assert(! isEmpty(domain) && length > 0);

   // The number of blocks in the x direction.
   const std::size_t numX = std::size_t(std::ceil((domain.upper[0] -
                                        domain.lower[0] +
                                        length / 2.) / length));
   // The number of blocks in the y direction.
   const std::size_t numY = std::size_t(std::ceil((domain.upper[1] -
                                        domain.lower[1] +
                                        length / 2.) / length));
   // The number of blocks in the z direction.
   const std::size_t num_z = std::size_t(std::ceil((domain.upper[2] -
                                         domain.lower[2] +
                                         length / 2.) / length));

   // Half length.
   const T hl = length / 2.0;

   //
   // Make the simplex set.
   //

   // The (un-indexed) simplex set.
   std::vector<Vertex> simplices;
   // The lower corner of the block.
   Vertex corner;
   // Axis
   Vertex a0, a1;
   //Ring
   std::array<Vertex, 4> r;
   // The simplex.
   Simplex s;
   // The centroid of the simplex.
   Vertex centroid;
   // Loop over the blocks.
   std::size_t x, y, z, n, m, mm;
   for (z = 0; z != num_z; ++z) {
      corner[2] = domain.lower[2] + z * length;
      for (y = 0; y != numY; ++y) {
         corner[1] = domain.lower[1] + y * length;
         for (x = 0; x != numX; ++x) {
            corner[0] = domain.lower[0] + x * length;

            // For each dimension.
            for (n = 0; n != 3; ++n) {
               std::size_t i = n;
               std::size_t j = (n + 1) % 3;
               std::size_t k = (n + 2) % 3;
               a1 = corner;
               a1 += hl;
               a0 = a1;
               a0[i] -= length;
               r[0] = corner;
               r[1] = r[0];
               r[1][j] += length;
               r[2] = r[1];
               r[2][k] += length;
               r[3] = r[2];
               r[3][j] -= length;

               // For the four simplices whose axis is in this direction.
               for (m = 0; m != 4; ++m) {
                  // Make the simplex.
                  s[0] = a0;
                  s[1] = a1;
                  s[2] = r[m];
                  s[3] = r[(m+1)%4];
                  // If the centroid is inside the object.
                  computeCentroid(s, &centroid);
                  if (f(centroid) <= 0) {
                     // Add the four vertices of the simplex to the set.
                     for (mm = 0; mm != 4; ++mm) {
                        simplices.push_back(s[mm]);
                     }
                  }
               }
            }
         }
      }
   }

   // Build the mesh from the simplex set.
   buildFromSimplices(simplices.begin(), simplices.end(), mesh);
}

} // namespace geom
}
