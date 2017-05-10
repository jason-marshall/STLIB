// -*- C++ -*-

#if !defined(__geom_mesh_iss_build_ipp__)
#error This file is an implementation detail.
#endif

namespace stlib
{
namespace geom {

// Build from a quadrilateral mesh.
template<std::size_t N, typename T>
inline
void
buildFromQuadMesh(const QuadMesh<N, T>& quadMesh,
                  IndSimpSet<N, 2, T>* mesh) {
   typedef typename QuadMesh<N, T>::IndexedFace IndexedFace;

   // The simplicial mesh will have the same vertices, but twice the number of
   // faces as the quad mesh.
   mesh->vertices = quadMesh.getVertices();
   mesh->indexedSimplices.resize(2 * quadMesh.getFacesSize());

   // Make two triangles from each quadrilateral.
   for (std::size_t faceIndex = 0; faceIndex != quadMesh.getFacesSize();
         ++faceIndex) {
      const T diagonal02 =
        ext::squaredDistance(quadMesh.getFaceVertex(faceIndex, 0),
                             quadMesh.getFaceVertex(faceIndex, 2));
      const T diagonal13 =
        ext::squaredDistance(quadMesh.getFaceVertex(faceIndex, 1),
                             quadMesh.getFaceVertex(faceIndex, 3));
      const IndexedFace& indexedFace = quadMesh.getIndexedFace(faceIndex);
      // Choose the smaller diagonal to split the quadrilateral.
      if (diagonal02 < diagonal13) {
         mesh->indexedSimplices[2 * faceIndex][0] = indexedFace[0];
         mesh->indexedSimplices[2 * faceIndex][1] = indexedFace[1];
         mesh->indexedSimplices[2 * faceIndex][2] = indexedFace[2];
         mesh->indexedSimplices[2 * faceIndex + 1][0] = indexedFace[2];
         mesh->indexedSimplices[2 * faceIndex + 1][1] = indexedFace[3];
         mesh->indexedSimplices[2 * faceIndex + 1][2] = indexedFace[0];
      }
      else {
         mesh->indexedSimplices[2 * faceIndex][0] = indexedFace[1];
         mesh->indexedSimplices[2 * faceIndex][1] = indexedFace[2];
         mesh->indexedSimplices[2 * faceIndex][2] = indexedFace[3];
         mesh->indexedSimplices[2 * faceIndex + 1][0] = indexedFace[3];
         mesh->indexedSimplices[2 * faceIndex + 1][1] = indexedFace[0];
         mesh->indexedSimplices[2 * faceIndex + 1][2] = indexedFace[1];
      }
   }

   // Update any auxilliary topological information.
   mesh->updateTopology();
}



template < std::size_t N, std::size_t M, typename T,
         typename IntForIter >
inline
void
buildFromSubsetVertices(const IndSimpSet<N, M, T>& in,
                        IntForIter verticesBeginning,
                        IntForIter verticesEnd,
                        IndSimpSet<N, M, T>* out) {
   typedef IndSimpSet<N, M, T> ISS;
   typedef typename ISS::IndexedSimplex IndexedSimplex;

   // Flag whether each vertex is in the subset.
   std::vector<bool> subset(in.vertices.size(), false);
   for (; verticesBeginning != verticesEnd; ++verticesBeginning) {
      subset[*verticesBeginning] = true;
   }

   // Determine the simplices that only use the subset of vertices.
   std::size_t m;
   std::vector<std::size_t> simplexIndices;
   for (std::size_t i = 0; i != in.indexedSimplices.size(); ++i) {
      const IndexedSimplex& s = in.indexedSimplices[i];
      // Check to see if each vertex is in the subset.
      for (m = 0; m != ISS::M + 1; ++m) {
         if (subset[s[m]] == false) {
            break;
         }
      }
      // If all vertices are in the subset.
      if (m == M + 1) {
         simplexIndices.push_back(i);
      }
   }

   // Make the mesh based on the subset of simplex indices.
   buildFromSubsetSimplices(in, simplexIndices.begin(), simplexIndices.end(),
                            out);
}



template < std::size_t N, std::size_t M, typename T,
         typename IntForIter >
inline
void
buildFromSubsetSimplices(const IndSimpSet<N, M, T>& in,
                         IntForIter simplicesBeginning,
                         IntForIter simplicesEnd,
                         IndSimpSet<N, M, T>* out) {
   typedef IndSimpSet<N, M, T> ISS;
   typedef typename ISS::IndexedSimplexIterator IndexedSimplexIterator;

   // Copy the subset of indexed simplices.
   const std::size_t numSimplices = std::distance(simplicesBeginning,
                                    simplicesEnd);
   out->indexedSimplices.resize(numSimplices);
   for (IndexedSimplexIterator s = out->indexedSimplices.begin();
         s != out->indexedSimplices.end(); ++s, ++simplicesBeginning) {
      // Make sure the simplex indices are valid.
      assert(*simplicesBeginning < in.indexedSimplices.size());
      // Copy the indexed simplex.
      *s = in.indexedSimplices[*simplicesBeginning];
   }

   // Copy the vertices.
   out->vertices = in.vertices;
   // Pack the mesh to get rid of unused vertices.
   pack(out);
}



template<std::size_t N, std::size_t M, typename T, class LSF>
inline
void
buildFromVerticesInside(const IndSimpSet<N, M, T>& in,
                        const LSF& f,
                        IndSimpSet<N, M, T>* out) {
   // Determine the vertex indices that are inside.
   std::vector<std::size_t> insideIndices;
   determineVerticesInside(in, f, std::back_inserter(insideIndices));
   // Select the portion of this mesh whose vertices are inside.
   buildFromSubsetVertices(in, insideIndices.begin(), insideIndices.end(), out);
}



template<std::size_t N, std::size_t M, typename T, class LSF>
inline
void
buildFromSimplicesInside(const IndSimpSet<N, M, T>& in,
                         const LSF& f,
                         IndSimpSet<N, M, T>* out) {
   // Determine the simplex indices that are inside.
   std::vector<std::size_t> insideIndices;
   determineSimplicesInside(in, f, std::back_inserter(insideIndices));
   // Select the portion of this mesh whose simplices are inside.
   buildFromSubsetSimplices(in, insideIndices.begin(), insideIndices.end(),
                            out);
}



template < std::size_t N, std::size_t M, typename T,
         typename IntOutputIterator >
inline
void
buildBoundary(const IndSimpSetIncAdj<N, M, T>& in,
              IndSimpSet < N, M - 1, T > * out,
              IntOutputIterator usedVertexIndices) {
   // Build the boundary without packing.
   buildBoundaryWithoutPacking(in, out);

   // Pack the mesh to get rid of the interior vertices.
   pack(out, usedVertexIndices);
}



template < std::size_t N, std::size_t M, typename T,
         typename IntOutputIterator >
inline
void
buildBoundaryWithoutPacking(const IndSimpSetIncAdj<N, M, T>& in,
                            IndSimpSet < N, M - 1, T > * out,
                            IntOutputIterator incidentSimplices) {
   typedef typename IndSimpSetIncAdj<N, M, T>::IndexedSimplexFace
   IndexedSimplexFace;

   // The set of indexed faces that lie on the boundary.
   std::vector<IndexedSimplexFace> boundaryFaces;

   std::size_t m;
   // Loop over the simplices.
   for (std::size_t n = 0; n != in.indexedSimplices.size(); ++n) {
      // Loop over the faces of the simplex.
      for (m = 0; m != M + 1; ++m) {
         // If the face is on the boundary.
         if (in.adjacent[n][m] == std::numeric_limits<std::size_t>::max()) {
            // Add the indexed face.
            boundaryFaces.push_back(getFace(in.indexedSimplices[n], m));
            // Record the incident simplex for the face.
            *incidentSimplices++ = n;
         }
      }
   }

   // Build the boundary with all of the vertices and the boundary faces.
   build(out, in.vertices, boundaryFaces);
}




// Make a mesh (separated into connected components) that is the boundary of
// the input mesh.
template < std::size_t N, std::size_t M, typename T,
         typename IntOutputIterator1, typename IntOutputIterator2 >
inline
void
buildBoundaryOfComponentsWithoutPacking
(const IndSimpSetIncAdj<N, M, T>& in,
 IndSimpSet < N, M - 1, T > * out,
 IntOutputIterator1 delimiterIterator,
 IntOutputIterator2 incidentSimplices) {
   // First get the boundary of the mesh.
   std::vector<std::size_t> incident;
   buildBoundaryWithoutPacking(in, out, std::back_inserter(incident));

   // Then separate the boundary into connected components.
   std::vector<std::size_t> permutation;
   {
      IndSimpSetIncAdj < N, M - 1, T > tmp(*out);
      separateComponents(&tmp, delimiterIterator,
                         std::back_inserter(permutation));
      *out = tmp;
   }

   // Permute the values for the incident simplices.
   for (std::vector<std::size_t>::const_iterator i = permutation.begin();
         i != permutation.end(); ++i) {
      *incidentSimplices++ = incident[*i];
   }
}



template<std::size_t N, typename T>
inline
void
centerPointMeshSetSimplices(const IndSimpSet<N, 1, T>& boundary,
                            IndSimpSet<N, 2, T>* mesh) {
   const std::size_t M = 2;
   std::size_t m;
   for (std::size_t n = 0; n != mesh->indexedSimplices.size(); ++n) {
      for (m = 0; m != M; ++m) {
         mesh->indexedSimplices[n][m] = boundary.indexedSimplices[n][m];
      }
      mesh->indexedSimplices[n][M] = mesh->vertices.size() - 1;
   }
}



template<std::size_t N, typename T>
inline
void
centerPointMeshSetSimplices(const IndSimpSet<N, 2, T>& boundary,
                            IndSimpSet<N, 3, T>* mesh) {
   const std::size_t M = 3;
   std::size_t m;
   for (std::size_t n = 0; n != mesh->indexedSimplices.size(); ++n) {
      mesh->indexedSimplices[n][0] = mesh->vertices.size() - 1;
      for (m = 0; m != M; ++m) {
         mesh->indexedSimplices[n][m+1] =
            boundary.indexedSimplices[n][m];
      }
   }
}



// Make a mesh by connecting the boundary nodes to a new center point.
template<std::size_t N, std::size_t M, typename T>
inline
void
centerPointMesh(const IndSimpSet < N, M - 1, T > & boundary,
                IndSimpSet<N, M, T>* mesh) {
   typedef IndSimpSet<N, M, T> ISS;
   typedef typename ISS::Vertex Vertex;

   // Sanity check.
   assert(boundary.vertices.size() != 0);
   assert(boundary.indexedSimplices.size() != 0);

   // Resize the mesh.
   mesh->vertices.resize(boundary.vertices.size() + 1);
   mesh->indexedSimplices.resize(boundary.indexedSimplices.size());

   // Set the vertices.
   {
      Vertex p = ext::filled_array<Vertex>(0.0);
      for (std::size_t n = 0; n != boundary.vertices.size(); ++n) {
         p += mesh->vertices[n] = boundary.vertices[n];
      }
      p /= T(boundary.vertices.size());
      mesh->vertices[mesh->vertices.size() - 1] = p;
   }

   // Set the indexed simplices.
   centerPointMeshSetSimplices(boundary, mesh);

   // Update any auxilliary topological information.
   mesh->updateTopology();
}



// Merge a range of meshes to make a single mesh.
template < std::size_t N, std::size_t M, typename T,
         typename MeshInputIterator >
inline
void
merge(MeshInputIterator beginning, MeshInputIterator end,
      IndSimpSet<N, M, T>* out) {
   typedef IndSimpSet<N, M, T> Mesh;
   typedef typename Mesh::Vertex Vertex;
   typedef typename Mesh::IndexedSimplex IndexedSimplex;

   std::vector<Vertex> vertices;
   std::vector<IndexedSimplex> indexedSimplices;

   std::size_t indexOffset = 0;
   IndexedSimplex s;
   // For each input mesh.
   for (; beginning != end; ++beginning) {
      // Get the vertices.
      for (std::size_t i = 0; i != beginning->vertices.size(); ++i) {
         vertices.push_back(beginning->vertices[i]);
      }
      // Get the indexed simplices.
      for (std::size_t i = 0; i != beginning->indexedSimplices.size(); ++i) {
         // Get the indexed simplex.
         s = beginning->indexedSimplices[i];
         // Offset the vertex indices.
         for (std::size_t m = 0; m != M + 1; ++m) {
            s[m] += indexOffset;
         }
         indexedSimplices.push_back(s);
      }
      // Update the index offset.
      indexOffset += beginning->vertices.size();
   }

   // Build the mesh.
   build(out, vertices.size(), &vertices[0], indexedSimplices.size(),
         &indexedSimplices[0]);
}



// Merge two meshes to make a single mesh.
template<std::size_t N, std::size_t M, typename T>
inline
void
merge2(const IndSimpSet<N, M, T>& a, const IndSimpSet<N, M, T>& b,
       IndSimpSet<N, M, T>* out) {
   typedef IndSimpSet<N, M, T> Mesh;

   // Make an array of the two meshes.
   std::array<const Mesh*, 2> meshes = {{&a, &b}};
   // Call the above merge function.
   merge(ads::constructIndirectIterator(meshes.begin()),
   ads::constructIndirectIterator(meshes.end()), out);
}

} // namespace geom
}
