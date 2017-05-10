// -*- C++ -*-

#if !defined(__geom_mesh_iss_subdivide_ipp__)
#error This file is an implementation detail.
#endif

namespace stlib
{
namespace geom {


// Subdivide by splitting each simplex in half.
template<std::size_t N, typename T>
inline
void
subdivide(const IndSimpSet<N, 1, T>& in,
          IndSimpSet<N, 1, T>* out) {
   // Resize the output mesh.
   build(out, in.vertices.size() + in.indexedSimplices.size(),
         2 * in.indexedSimplices.size());

   //
   // Build the vertices.
   //

   // The old vertices.
   std::size_t vertexIndex = 0;
   for (std::size_t i = 0; i != in.vertices.size(); ++i, ++vertexIndex) {
      out->vertices[vertexIndex] = in.vertices[i];
   }
   // The midpoints.
   for (std::size_t i = 0; i != in.indexedSimplices.size(); ++i, ++vertexIndex) {
      out->vertices[vertexIndex] = in.getSimplexVertex(i, 0);
      out->vertices[vertexIndex] += in.getSimplexVertex(i, 1);
      out->vertices[vertexIndex] /= 2.0;
   }

   //
   // Build the indexed simplices.
   //
   for (std::size_t i = 0; i != in.indexedSimplices.size(); ++i) {
      out->indexedSimplices[2 * i][0] = in.indexedSimplices[i][0];
      out->indexedSimplices[2 * i][1] = in.vertices.size() + i;
      out->indexedSimplices[2 * i + 1][0] = in.vertices.size() + i;
      out->indexedSimplices[2 * i + 1][1] = in.indexedSimplices[i][1];
   }

   // Update the topology if necessary.
   out->updateTopology();
}



// Subdivide by splitting each simplex into four similar simplices.
template<std::size_t N, typename T>
inline
void
subdivide(const IndSimpSetIncAdj<N, 2, T>& in,
          IndSimpSet<N, 2, T>* out) {
   typedef IndSimpSet<N, 2, T> OutputMesh;
   typedef typename OutputMesh::Vertex Vertex;
   typedef typename OutputMesh::IndexedSimplex IndexedSimplex;

   const std::size_t M = 2;

   // The vertices in the subdivided mesh.
   std::vector<Vertex> vertices;
   // The indexed midpoint vertices for each simplex.
   std::vector<IndexedSimplex> 
      midpointIndexedSimplices(in.indexedSimplices.size());

   // Add the vertices from the input mesh.
   for (std::size_t i = 0; i != in.vertices.size(); ++i) {
      vertices.push_back(in.vertices[i]);
   }

   //
   // Loop over the simplices, determining the midpoint vertices.
   //
   std::size_t adjacentSimplex;
   // For each simplex.
   for (std::size_t i = 0; i != in.indexedSimplices.size(); ++i) {
      // For each vertex of the simplex.
      for (std::size_t m = 0; m != M + 1; ++m) {
         adjacentSimplex = in.adjacent[i][m];
         // If there is no adjacent simplex or if this simplices' index is less
         // than the adjacent simplices' index.
         if (adjacentSimplex == std::size_t(-1) || i < adjacentSimplex) {
            // Record the index of the new midpoint vertex.
            midpointIndexedSimplices[i][m] = vertices.size();
            // Make a new midpoint vertex.
            vertices.push_back((in.getSimplexVertex(i, (m + 1) % (M + 1)) +
                                in.getSimplexVertex(i, (m + 2) % (M + 1))) /
                               T(2.0));
         }
         // Otherwise, the adjacent simplex already added the midpoint vertex.
         else {
            // Record the index of that midpoint vertex.
            midpointIndexedSimplices[i][m] =
               midpointIndexedSimplices[adjacentSimplex]
               [in.getMirrorIndex(i, m)];
         }
      }
   }


   // Resize the output mesh.
   build(out, vertices.size(), 4 * in.indexedSimplices.size());

   // Set the vertices.
   for (std::size_t i = 0; i != out->vertices.size(); ++i) {
      out->vertices[i] = vertices[i];
   }

   //
   // Build the indexed simplices in the output mesh from the input simplices
   // and midpoint simplices.
   //
   for (std::size_t i = 0; i != in.indexedSimplices.size(); ++i) {
      out->indexedSimplices[4 * i][0] = in.indexedSimplices[i][0];
      out->indexedSimplices[4 * i][1] = midpointIndexedSimplices[i][2];
      out->indexedSimplices[4 * i][2] = midpointIndexedSimplices[i][1];

      out->indexedSimplices[4 * i + 1][0] = in.indexedSimplices[i][1];
      out->indexedSimplices[4 * i + 1][1] = midpointIndexedSimplices[i][0];
      out->indexedSimplices[4 * i + 1][2] = midpointIndexedSimplices[i][2];

      out->indexedSimplices[4 * i + 2][0] = in.indexedSimplices[i][2];
      out->indexedSimplices[4 * i + 2][1] = midpointIndexedSimplices[i][1];
      out->indexedSimplices[4 * i + 2][2] = midpointIndexedSimplices[i][0];

      out->indexedSimplices[4 * i + 3][0] = midpointIndexedSimplices[i][0];
      out->indexedSimplices[4 * i + 3][1] = midpointIndexedSimplices[i][1];
      out->indexedSimplices[4 * i + 3][2] = midpointIndexedSimplices[i][2];
   }

   // Update the topology if necessary.
   out->updateTopology();
}

// Subdivide by splitting each simplex into eight simplices.
template<std::size_t N, typename T>
inline
void
subdivide(const IndSimpSetIncAdj<N, 3, T>& in, IndSimpSet<N, 3, T>* out) {
   typedef IndSimpSet<N, 3, T> OutputMesh;
   typedef typename OutputMesh::Vertex Vertex;
   typedef typename OutputMesh::IndexedSimplex IndexedSimplex;
   typedef std::map<std::size_t, std::size_t> Map;
   typedef container::TriangularArray<std::size_t, container::UpperTriangular,
                                      container::StrictlyTriangular>
      TriangularArray;

   const std::size_t M = 3;
   // The number of vertices in the input mesh.
   const std::size_t V = in.vertices.size();
   // We use keys to label the midpoint vertex indices. Make sure that 
   // std::size_t can be used to store the keys.
   // (V - 1) * V + (V-1) < max
   // V^2 - 1 < max
   assert(V < 
          std::size_t(1) << (std::numeric_limits<std::size_t>::digits / 2 + 1));

   // The vertices in the subdivided mesh.
   // Start with the vertices from the input mesh.
   std::vector<Vertex> vertices = in.vertices;

   //
   // Loop over the simplices, determining the midpoint vertices.
   //
   Map midpointIndices;
   // For each simplex.
   for (std::size_t i = 0; i != in.indexedSimplices.size(); ++i) {
      // For each edge of the simplex.
      for (std::size_t m = 0; m != M + 1; ++m) {
         for (std::size_t n = m + 1; n != M + 1; ++n) {
            std::size_t a = in.indexedSimplices[i][m];
            std::size_t b = in.indexedSimplices[i][n];
            if (b < a) {
               std::swap(a, b);
            }
            std::size_t key = a * V + b;
            if (! midpointIndices.count(key)) {
               // Record the index of the midpoint vertex, which we will access
               // using the key.
               midpointIndices.insert(std::pair<const std::size_t, std::size_t>
                                      (key, vertices.size()));
               // Record the location of the midpoint vertex.
               vertices.push_back(T(0.5) * (in.vertices[a] + in.vertices[b]));
            }
         }
      }
   }

   std::vector<IndexedSimplex> indexedSimplices(8 * in.indexedSimplices.size());

   //
   // Build the indexed simplices in the output mesh from the input simplices
   // and midpoint simplices.
   //
   // The corner vertex indices.
   IndexedSimplex v;
   // The midpoint vertex indices.
   TriangularArray m(4);
   std::size_t key;
   T d1, d2, d3, minD;
   std::array<std::size_t, 6> oct;
   for (std::size_t i = 0; i != in.indexedSimplices.size(); ++i) {
      // Record the vertex indices.
      v = in.indexedSimplices[i];
      // Record the midpoint vertex indices.
      for (std::size_t j = 0; j != M + 1; ++j) {
         for (std::size_t k = j + 1; k != M + 1; ++k) {
            if (v[j] < v[k]) {
               key = v[j] * V + v[k];
            }
            else {
               key = v[k] * V + v[j];
            }
            m(j, k) = midpointIndices[key];
         }
      }
      //
      // The corner tets.
      //
      indexedSimplices[8*i][0] = v[0];
      indexedSimplices[8*i][1] = m(0, 1);
      indexedSimplices[8*i][2] = m(0, 2);
      indexedSimplices[8*i][3] = m(0, 3);

      indexedSimplices[8*i+1][0] = v[1];
      indexedSimplices[8*i+1][1] = m(0, 1);
      indexedSimplices[8*i+1][2] = m(1, 3);
      indexedSimplices[8*i+1][3] = m(1, 2);

      indexedSimplices[8*i+2][0] = v[2];
      indexedSimplices[8*i+2][1] = m(0, 2);
      indexedSimplices[8*i+2][2] = m(1, 2);
      indexedSimplices[8*i+2][3] = m(2, 3);

      indexedSimplices[8*i+3][0] = v[3];
      indexedSimplices[8*i+3][1] = m(0, 3);
      indexedSimplices[8*i+3][2] = m(2, 3);
      indexedSimplices[8*i+3][3] = m(1, 3);

      // Diagonal that uses the midpoint (0, n).
      d1 = ext::squaredDistance(vertices[m(0, 1)], vertices[m(2, 3)]);
      d2 = ext::squaredDistance(vertices[m(0, 2)], vertices[m(1, 3)]);
      d3 = ext::squaredDistance(vertices[m(0, 3)], vertices[m(1, 2)]);
      minD = std::min(std::min(d1, d2), d3);

      if (d1 == minD) {
         // Bottom.
         oct[0] = m(0, 1);
         // Top.
         oct[1] = m(2, 3);
         // Ring.
         oct[2] = m(1, 3);
         oct[3] = m(1, 2);
         oct[4] = m(0, 2);
         oct[5] = m(0, 3);
      }
      else if (d2 == minD) {
         // Bottom.
         oct[0] = m(0, 2);
         // Top.
         oct[1] = m(1, 3);
         // Ring.
         oct[2] = m(0, 1);
         oct[3] = m(1, 2);
         oct[4] = m(2, 3);
         oct[5] = m(0, 3);
      }
      else {
         // Bottom.
         oct[0] = m(0, 3);
         // Top.
         oct[1] = m(1, 2);
         // Ring.
         oct[2] = m(0, 1);
         oct[3] = m(0, 2);
         oct[4] = m(2, 3);
         oct[5] = m(1, 3);
      }
      //
      // The midpoint tets.
      //
      for (std::size_t j = 0; j != 4; ++j) {
         indexedSimplices[8*i+4+j][0] = oct[0];
         indexedSimplices[8*i+4+j][1] = oct[1];
         indexedSimplices[8*i+4+j][2] = oct[2 + j];
         indexedSimplices[8*i+4+j][3] = oct[2 + (j + 1) % 4];
      }
   }

   // Set the vertices and indexed simplices.
   out->vertices.swap(vertices);
   out->indexedSimplices.swap(indexedSimplices);
   // Update the topology if necessary.
   out->updateTopology();
}

} // namespace geom
}
