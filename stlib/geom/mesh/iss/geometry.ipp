// -*- C++ -*-

#if !defined(__geom_mesh_iss_geometry_ipp__)
#error This file is an implementation detail.
#endif

namespace stlib
{
namespace geom {

// Return the outward normal at the specified vertex.
template<typename T>
inline
typename IndSimpSetIncAdj<2, 1, T>::Vertex
computeVertexNormal(const IndSimpSetIncAdj<2, 1, T>& mesh, const std::size_t n) {
   typedef IndSimpSetIncAdj<2, 1, T> ISS;
   typedef typename ISS::Vertex Vertex;

   Vertex x;
   computeVertexNormal(mesh, n, &x);
   return x;
}

// Compute the outward normal at the specified vertex.
template<typename T>
inline
void
computeVertexNormal(const IndSimpSetIncAdj<2, 1, T>& mesh, const std::size_t n,
                    typename IndSimpSetIncAdj<2, 1, T>::Vertex*
                    vertexNormal) {
   typedef IndSimpSetIncAdj<2, 1, T> ISS;
   typedef typename ISS::Vertex Vertex;

   // The vertex should have two incident faces.
   assert(mesh.incident.size(n) == 2);

   std::fill(vertexNormal->begin(), vertexNormal->end(), 0.0);

   // The first incident edge.
   std::size_t i = mesh.incident(n, 0);
   Vertex a = mesh.vertices[mesh.indexedSimplices[i][1]];
   a -= mesh.vertices[mesh.indexedSimplices[i][0]];
   rotateMinusPiOver2(&a);
   ext::normalize(&a);
   *vertexNormal += a;

   // The second incident edge.
   i = mesh.incident(n, 1);
   a = mesh.vertices[mesh.indexedSimplices[i][1]];
   a -= mesh.vertices[mesh.indexedSimplices[i][0]];
   rotateMinusPiOver2(&a);
   ext::normalize(&a);
   *vertexNormal += a;

   ext::normalize(vertexNormal);
}



// Return the outward normal at the specified vertex.
template<typename T>
inline
typename IndSimpSetIncAdj<2, 2, T>::Vertex
computeVertexNormal(const IndSimpSetIncAdj<2, 2, T>& mesh, const std::size_t n) {
   typedef IndSimpSetIncAdj<2, 2, T> ISS;
   typedef typename ISS::Vertex Vertex;

   Vertex x;
   computeVertexNormal(mesh, n, &x);
   return x;
}



// Compute the outward normal at the specified vertex.
template<typename T>
inline
void
computeVertexNormal(const IndSimpSetIncAdj<2, 2, T>& mesh, const std::size_t n,
                    typename IndSimpSetIncAdj<2, 2, T>::Vertex*
                    vertexNormal) {
   typedef IndSimpSetIncAdj<2, 2, T> ISS;
   typedef typename ISS::Vertex Vertex;
   typedef typename ISS::IndexedSimplex IndexedSimplex;
   typedef typename ISS::IncidenceConstIterator IncidenceConstIterator;

#ifdef STLIB_DEBUG
   // This should be a boundary vertex.
   assert(mesh.isVertexOnBoundary(n));
   // The vertex should have two incident faces.
   assert(mesh.getIncidentSize(n) >= 1);
#endif

   const std::size_t M = 2;

   std::size_t i;
   Vertex v;

   std::fill(vertexNormal->begin(), vertexNormal->end(), 0.0);
   // For each incident simplex.
   const IncidenceConstIterator iterEnd = mesh.incident.end(n);
   for (IncidenceConstIterator iter = mesh.incident.begin(n);
         iter != iterEnd; ++iter) {
      // The indexed simplex that defines the face.
      const IndexedSimplex& is = mesh.indexedSimplices[*iter];
      // The local index of the n_th vertex in the face.
      i = ext::index(is, n);

      if (mesh.adjacent[*iter][(i + 2) % (M + 1)] ==
          std::numeric_limits<std::size_t>::max()) {
         v = mesh.vertices[is[(i+1)%(M+1)]];
         v -= mesh.vertices[n];
         rotateMinusPiOver2(&v);
         ext::normalize(&v);
         *vertexNormal += v;
      }

      if (mesh.adjacent[*iter][(i + 1) % (M + 1)] == std::size_t(-1)) {
         v = mesh.vertices[is[(i+2)%(M+1)]];
         v -= mesh.vertices[n];
         rotatePiOver2(&v);
         ext::normalize(&v);
         *vertexNormal += v;
      }
   }
   ext::normalize(vertexNormal);
}



// Return the outward normal at the specified vertex.
template<typename T>
inline
typename IndSimpSetIncAdj<3, 2, T>::Vertex
computeVertexNormal(const IndSimpSetIncAdj<3, 2, T>& mesh, const std::size_t n) {
   typedef IndSimpSetIncAdj<3, 2, T> ISS;
   typedef typename ISS::Vertex Vertex;

   Vertex x;
   computeVertexNormal(mesh, n, &x);
   return x;
}



// Compute the outward normal at the specified vertex.
template<typename T>
inline
void
computeVertexNormal(const IndSimpSetIncAdj<3, 2, T>& mesh, const std::size_t n,
                    typename IndSimpSetIncAdj<3, 2, T>::Vertex*
                    vertexNormal) {
   typedef IndSimpSetIncAdj<3, 2, T> ISS;
   typedef typename ISS::Vertex Vertex;
   typedef typename ISS::IndexedSimplex IndexedSimplex;
   typedef typename ISS::IncidenceConstIterator IncidenceConstIterator;

   // The vertex should have at least 3 incident faces.
   assert(mesh.incident.size(n) >= 3);

   std::fill(vertexNormal->begin(), vertexNormal->end(), 0.0);
   std::size_t i;
   // CONTINUE
   //Simplex s;
   Vertex x, y, faceNormal;

   // For each incident face.
   const IncidenceConstIterator iterEnd = mesh.incident.end(n);
   for (IncidenceConstIterator iter = mesh.incident.begin(n);
         iter != iterEnd; ++iter) {
      // The indexed simplex that defines the face.
      const IndexedSimplex& is = mesh.indexedSimplices[*iter];
      // The local index of the n_th vertex in the face.
      i = ext::index(is, n);
      // Construct the face.
      // CONTINUE
      /*
      s[0] = mesh.vertices[is[i]];
      s[1] = mesh.vertices[is[(i+1)%3]];
      s[2] = mesh.vertices[is[(i+2)%3]];
      */

      // Compute the face normal.
      x = mesh.vertices[is[(i+1)%3]];
      x -= mesh.vertices[is[i]];
      ext::normalize(&x);
      y = mesh.vertices[is[(i+2)%3]];
      y -= mesh.vertices[is[i]];
      ext::normalize(&y);
      ext::cross(x, y, &faceNormal);
      ext::normalize(&faceNormal);

      // Contribute to the vertex normal.
      // Multiply by the angle between the edges.
      faceNormal *= std::acos(ext::dot(x, y));
      *vertexNormal += faceNormal;
   }
   ext::normalize(vertexNormal);
}



// Return the outward normal at the specified boundary vertex.
template<typename T>
inline
typename IndSimpSetIncAdj<3, 3, T>::Vertex
computeVertexNormal(const IndSimpSetIncAdj<3, 3, T>& mesh, const std::size_t n) {
   typedef IndSimpSetIncAdj<3, 3, T> ISS;
   typedef typename ISS::Vertex Vertex;

   Vertex x;
   computeVertexNormal(mesh, n, &x);
   return x;
}



// Compute the outward normal at the specified boundary vertex.
template<typename T>
inline
void
computeVertexNormal(const IndSimpSetIncAdj<3, 3, T>& mesh, const std::size_t n,
                    typename IndSimpSetIncAdj<3, 3, T>::Vertex*
                    vertexNormal) {
   typedef IndSimpSetIncAdj<3, 3, T> ISS;
   typedef typename ISS::Vertex Vertex;
   typedef typename ISS::IndexedSimplex IndexedSimplex;
   typedef typename ISS::IncidenceConstIterator IncidenceConstIterator;
   typedef typename ISS::IndexedSimplexFace IndexedSimplexFace;

#ifdef STLIB_DEBUG
   // This should be a boundary vertex.
   assert(mesh.isVertexOnBoundary(n));
   // The vertex should have two incident faces.
   assert(mesh.getIncidentSize(n) >= 1);
#endif

   const std::size_t M = 3;

   std::size_t i, j;
   Vertex x, y, faceNormal;
   IndexedSimplexFace face;

   std::fill(vertexNormal->begin(), vertexNormal->end(), 0.0);
   // For each incident simplex.
   const IncidenceConstIterator iterEnd = mesh.incident.end(n);
   for (IncidenceConstIterator iter = mesh.incident.begin(n);
         iter != iterEnd; ++iter) {
      // The incident indexed simplex.
      const IndexedSimplex& is = mesh.indexedSimplices[*iter];
      // The local index of the n_th vertex in the incident simplex.
      i = ext::index(is, n);

      // Loop over the incident faces.
      for (j = 1; j != M + 1; ++j) {
         // If this is a boundary face.
         if (mesh.adjacent[*iter][(i + j) % (M + 1)] == std::size_t(-1)) {
            // Get the indexed face.
            getFace(is, (i + j) % (M + 1), &face);

            // Compute the face normal.
            x = mesh.vertices[face[1]];
            x -= mesh.vertices[face[0]];
            ext::normalize(&x);
            y = mesh.vertices[face[2]];
            y -= mesh.vertices[face[0]];
            ext::normalize(&y);
            ext::cross(x, y, &faceNormal);
            ext::normalize(&faceNormal);

            // Contribute to the vertex normal.
            // Multiply by the angle between the edges.
            faceNormal *= std::acos(ext::dot(x, y));
            *vertexNormal += faceNormal;
         }
      }
   }
   ext::normalize(vertexNormal);
}








// Compute the outward normal for the specified simplex (triangle face).
template<typename T>
inline
void
computeSimplexNormal(const IndSimpSetIncAdj<3, 2, T>& mesh,
                     const std::size_t simplexIndex,
                     typename IndSimpSetIncAdj<3, 2, T>::Vertex* simplexNormal) {
   typedef IndSimpSetIncAdj<3, 2, T> ISS;
   typedef typename ISS::Vertex Vertex;

   assert(simplexIndex < mesh.indexedSimplices.size());

   Vertex x, y;
   // Compute the face normal.
   x = mesh.getSimplexVertex(simplexIndex, 2);
   x -= mesh.getSimplexVertex(simplexIndex, 1);
   y = mesh.getSimplexVertex(simplexIndex, 0);
   y -= mesh.getSimplexVertex(simplexIndex, 1);
   ext::cross(x, y, simplexNormal);
   ext::normalize(simplexNormal);
}




// Compute the outward normals for the simplices (triangle faces).
template<typename T>
inline
void
computeSimplexNormals
(const IndSimpSetIncAdj<3, 2, T>& mesh,
 std::vector<typename IndSimpSetIncAdj<3, 2, T>::Vertex>* simplexNormals) {
   typedef IndSimpSetIncAdj<3, 2, T> ISS;
   typedef typename ISS::Simplex Simplex;
   typedef typename ISS::Vertex Vertex;

   assert(mesh.indexedSimplices.size() == simplexNormals->size());

   Simplex simplex;
   Vertex normal, x, y;
   // For each simplex.
   for (std::size_t n = 0; n != mesh.indexedSimplices.size(); ++n) {
      // Get the simplex.
      mesh.getSimplex(n, &simplex);
      // Compute the face normal.
      x = simplex[2];
      x -= simplex[1];
      y = simplex[0];
      y -= simplex[1];
      ext::cross(x, y, &normal);
      ext::normalize(&normal);
      (*simplexNormals)[n] = normal;
   }
}





// Compute the outward normals for the simplices (line segments).
template<typename T>
inline
void
computeSimplexNormals
(const IndSimpSetIncAdj<2, 1, T>& mesh,
 std::vector<typename IndSimpSetIncAdj<2, 1, T>::Vertex>* simplexNormals) {
   typedef IndSimpSetIncAdj<2, 1, T> ISS;
   typedef typename ISS::Vertex Vertex;

   assert(mesh.indexedSimplices.size() == simplexNormals->size());

   Vertex x;
   // For each simplex.
   for (std::size_t n = 0; n != mesh.indexedSimplices.size(); ++n) {
      // The tangent direction.
      x = mesh.getSimplexVertex(n, 1);
      x -= mesh.getSimplexVertex(n, 0);
      // The normal direction.
      rotateMinusPiOver2(&x);
      // Normalize to unit length.
      ext::normalize(&x);
      (*simplexNormals)[n] = x;
   }
}



// Compute the outward normals for the vertices.
template<typename T>
inline
void
computeVertexNormals
(const IndSimpSetIncAdj<3, 2, T>& mesh,
 const std::vector<typename IndSimpSetIncAdj<3, 2, T>::Vertex>& simplexNormals,
 std::vector<typename IndSimpSetIncAdj<3, 2, T>::Vertex>* vertexNormals) {
   typedef IndSimpSetIncAdj<3, 2, T> ISS;
   typedef typename ISS::Simplex Simplex;
   typedef typename ISS::Vertex Vertex;

   assert(mesh.indexedSimplices.size() == simplexNormals.size() &&
          mesh.vertices.size() == vertexNormals->size());

   std::fill(vertexNormals->begin(), vertexNormals->end(),
             Vertex{{0., 0., 0.}});
   Vertex normal, x, y;
   Simplex simplex;
   std::size_t i, m;
   // For each simplex.
   for (std::size_t n = 0; n != mesh.indexedSimplices.size(); ++n) {
      // Get the simplex.
      mesh.getSimplex(n, &simplex);
      // Contribute to the vertex normal.
      for (m = 0; m != 3; ++m) {
         x = simplex[(m+1)%3];
         x -= simplex[m];
         ext::normalize(&x);
         y = simplex[(m+2)%3];
         y -= simplex[m];
         ext::normalize(&y);
         i = mesh.indexedSimplices[n][m];
         // Get the simplex normal.
         normal = simplexNormals[n];
         // Multiply by the angle between the edges.
         normal *= std::acos(ext::dot(x, y));
         (*vertexNormals)[i] += normal;
      }
   }

   // Normalize the vertex directions.
   for (std::size_t n = 0; n != vertexNormals->size(); ++n) {
     ext::normalize(&(*vertexNormals)[n]);
   }
}





// Compute the outward normals for the simplices and vertices.
template<typename T>
inline
void
computeSimplexAndVertexNormals
(const IndSimpSetIncAdj<3, 2, T>& mesh,
 std::vector<typename IndSimpSetIncAdj<3, 2, T>::Vertex>* simplexNormals,
 std::vector<typename IndSimpSetIncAdj<3, 2, T>::Vertex>* vertexNormals) {
   computeSimplexNormals(mesh, simplexNormals);
   computeVertexNormals(mesh, *simplexNormals, vertexNormals);
}





// Return the cosine of the interior angle at the specified vertex.
template<std::size_t N, typename T>
inline
T
computeCosineAngle(const IndSimpSetIncAdj<N, 1, T>& mesh,
                   const std::size_t vertexIndex) {
   typedef IndSimpSetIncAdj<N, 1, T> ISS;
   typedef typename ISS::Vertex Vertex;

   // The vertex should have two incident faces.
   assert(mesh.incident.size(vertexIndex) == 2);

   // The two incident edges.
   std::size_t i = mesh.incident(vertexIndex, 0);
   std::size_t j = mesh.incident(vertexIndex, 1);
   if (mesh.indexedSimplices[i][0] == vertexIndex) {
      std::swap(i, j);
   }
   assert(mesh.indexedSimplices[i][1] == vertexIndex &&
          mesh.indexedSimplices[j][0] == vertexIndex);
   // Now i is the previous edge and j is the next edge.

   // Make two unit vectors with tails at the n_th vertex and heads in the
   // directions of the neighboring vertices.
   Vertex a = mesh.getSimplexVertex(i, 0);
   a -= mesh.vertices[vertexIndex];
   ext::normalize(&a);
   Vertex b = mesh.getSimplexVertex(j, 1);
   b -= mesh.vertices[vertexIndex];
   ext::normalize(&b);

   // Return the cosine of the interior angle.
   return ext::dot(a, b);
}



// Return the cosine of the interior angle at the specified 1-face.
template<typename T>
inline
T
computeCosineAngle(const IndSimpSetIncAdj<3, 2, T>& mesh,
                   const typename IndSimpSetIncAdj<3, 2, T>::Face& face) {
   typedef IndSimpSetIncAdj<3, 2, T> ISS;
   typedef typename ISS::Vertex Vertex;

   // Check that the face is valid.
   assert(face.first < mesh.indexedSimplices.size());
   assert(face.second < 2 + 1);
   // It must be an internal face.
   assert(! mesh.isOnBoundary(face));

   // The cosine of the angle is the negative of the dot product of the
   // incident simplex normals.
   // n0 . n1 == cos(pi - a) == - cos(a)
   Vertex n0, n1;
   computeSimplexNormal(mesh, face.first, &n0);
   computeSimplexNormal(mesh, mesh.adjacent[face.first][face.second], &n1);
   return - ext::dot(n0, n1);
}



// Return the cosine of the interior angle at the specified boundary vertex.
template<typename T>
inline
T
computeCosineBoundaryAngle(const IndSimpSetIncAdj<3, 2, T>& mesh,
                           const std::size_t vertexIndex) {
   typedef IndSimpSetIncAdj<3, 2, T> ISS;
   typedef typename ISS::Vertex Vertex;
   typedef typename ISS::Face Face;

   // The simplex dimension.
   const std::size_t M = 2;

   // It should be a boundary vertex.
   assert(mesh.isVertexOnBoundary(vertexIndex));

   //
   // Get the two neighboring boundary vertices.
   //
   std::size_t neighbors[2];
   {
      std::size_t neighborCount = 0;
      std::size_t simplexIndex;
      std::size_t i1, i2;
      // For each incident simplex.
      for (std::size_t n = 0; n != mesh.incident.size(vertexIndex); ++n) {
         // The index of the simplex.
         simplexIndex = mesh.incident(vertexIndex, n);
         // For each face of the simplex.
         for (std::size_t m = 0; m != M + 1; ++m) {
            // Skip the face that is opposite the specified vertex.
            if (mesh.indexedSimplices[simplexIndex][m] != vertexIndex) {
               // If this is a boundary face.
               if (mesh.isOnBoundary(Face(simplexIndex, m))) {
                  // The vertex (other than the specified one) is a neighboring
                  // boundary vertex.
                  i1 = mesh.indexedSimplices[simplexIndex][(m + 1) % (M + 1)];
                  i2 = mesh.indexedSimplices[simplexIndex][(m + 2) % (M + 1)];
                  assert(neighborCount != 2);
                  if (i1 != vertexIndex) {
                     neighbors[neighborCount++] = i1;
                  }
                  else if (i2 != vertexIndex) {
                     neighbors[neighborCount++] = i2;
                  }
                  else {
                     assert(false);
                  }
               }
            }
         }
      }
      assert(neighborCount == 2);
   }

   // Make two unit vectors with tails at the specified vertex and heads in the
   // directions of the neighboring vertices.
   Vertex a = mesh.vertices[neighbors[0]];
   a -= mesh.vertices[vertexIndex];
   ext::normalize(&a);
   Vertex b = mesh.vertices[neighbors[1]];
   b -= mesh.vertices[vertexIndex];
   ext::normalize(&b);

   // Return the cosine of the interior angle.
   return ext::dot(a, b);
}



// Return the solid interior angle at the specified vertex.
template<typename T>
inline
T
computeAngle(const IndSimpSetIncAdj<3, 2, T>& mesh, const std::size_t n) {
   typedef IndSimpSetIncAdj<3, 2, T> ISS;
   typedef typename ISS::Vertex Vertex;
   typedef typename ISS::IndexedSimplex IndexedSimplex;
   typedef typename ISS::IncidenceConstIterator IncidenceConstIterator;
   typedef std::array < Vertex, 3 + 1 > Tetrahedron;

   // Get the inward pointing normal.
   Vertex inwardNormal;
   computeVertexNormal(mesh, n, &inwardNormal);
   ext::negateElements(&inwardNormal);


   std::size_t i;
   Tetrahedron t;
   T solidAngle = 0;

   // For each incident face.
   const IncidenceConstIterator iterEnd = mesh.incident.end(n);
   for (IncidenceConstIterator iter = mesh.incident.begin(n);
         iter != iterEnd; ++iter) {
      // The indexed simplex that defines the face.
      const IndexedSimplex& is = mesh.indexedSimplices[*iter];
      // The local index of the n_th vertex in the face.
      i = ext::index(is, n);
      // Construct the tetrahedron defined by the inward normal and the face.
      t[0] = mesh.vertices[is[i]];
      t[0] += inwardNormal;
      t[1] = mesh.vertices[is[i]];
      t[2] = mesh.vertices[is[(i+1)%3]];
      t[3] = mesh.vertices[is[(i+2)%3]];

      solidAngle += computeAngle(t, 1);
   }
   return solidAngle;
}

// Return the sum of the incident angles at the specified vertex.
template<typename T>
inline
T
computeAngle(const IndSimpSetIncAdj<3, 3, T>& mesh, std::size_t n) {
   const std::size_t N = 3;
   // The angle for interior vertices is 4 pi.
   if (! mesh.isVertexOnBoundary(n)) {
      return 4. * numerical::Constants<T>::Pi();
   }
   // For boundary vertices we must calculate the angle.
   typename IndSimpSetIncAdj<N, N, T>::Simplex s;
   T angle = 0;
   // For each incident simplex.
   for (std::size_t i = 0; i != mesh.incident.size(n); ++i) {
      const std::size_t simplexIndex = mesh.incident(n, i);
      const std::size_t vertexIndex =
        ext::index(mesh.indexedSimplices[simplexIndex], n);
#ifdef STLIB_DEBUG
      assert(vertexIndex < N + 1);
#endif
      mesh.getSimplex(simplexIndex, &s);
      angle += computeAngle(s, vertexIndex);
   }
   return angle;
}

// Return the sum of the incident angles at the specified vertex.
template<typename T>
inline
T
computeAngle(const IndSimpSetIncAdj<2, 2, T>& mesh, std::size_t n) {
   const std::size_t N = 2;
   // The angle for interior vertices is 2 pi.
   if (! mesh.isVertexOnBoundary(n)) {
      return 2. * numerical::Constants<T>::Pi();
   }
   // For boundary vertices we must calculate the angle.
   typename IndSimpSetIncAdj<N, N, T>::Simplex s;
   T angle = 0;
   // For each incident simplex.
   for (std::size_t i = 0; i != mesh.incident.size(n); ++i) {
      const std::size_t simplexIndex = mesh.incident(n, i);
      const std::size_t vertexIndex =
        ext::index(mesh.indexedSimplices[simplexIndex], n);
#ifdef STLIB_DEBUG
      assert(vertexIndex < N + 1);
#endif
      mesh.getSimplex(simplexIndex, &s);
      angle += computeAngle(s, vertexIndex);
   }
   return angle;
}

// Return the sum of the incident angles at the specified vertex.
template<typename T>
inline
T
computeAngle(const IndSimpSetIncAdj<1, 1, T>& mesh, std::size_t n) {
   if (! mesh.isVertexOnBoundary(n)) {
      return 2.;
   }
   return 1.;
}



// Project the line segments to 1-D and collect them.
template<typename T, typename OutputIterator>
inline
void
projectAndGetSimplices(const IndSimpSet<2, 1, T>& mesh,
                       OutputIterator simplices) {
   typedef IndSimpSet<2, 1, T> ISS;
   typedef std::array < std::array<T, 1>, 1 + 1 > Segment;
   typedef typename ISS::Simplex Simplex;
   Simplex s;
   Segment t;

   const std::size_t size = mesh.indexedSimplices.size();
   // For each simplex.
   for (std::size_t n = 0; n != size; ++n) {
      // Get the n_th simplex.
      mesh.getSimplex(n, &s);
      // Project the line segment in 2-D to a line segment in 1-D.
      projectToLowerDimension(s, &t);
      // Add the line segment to the sequence of simplices.
      *simplices++ = t;
   }
}



// Project the triangle simplices to 2-D and collect them.
template<typename T, typename OutputIterator>
inline
void
projectAndGetSimplices(const IndSimpSet<3, 2, T>& mesh,
                       OutputIterator simplices) {
   typedef IndSimpSet<3, 2, T> ISS;
   typedef std::array < std::array<T, 2>, 2 + 1 > Triangle;
   typedef typename ISS::Simplex Simplex;
   Simplex s;
   Triangle t;

   const std::size_t size = mesh.indexedSimplices.size();
   // For each simplex.
   for (std::size_t n = 0; n != size; ++n) {
      // Get the n_th simplex.
      mesh.getSimplex(n, &s);
      // Project the triangle in 3-D to a triangle in 2-D.
      projectToLowerDimension(s, &t);
      // Add the triangle to the sequence of simplices.
      *simplices++ = t;
   }
}

} // namespace geom
}
