// -*- C++ -*-

/*!
  \file geom/mesh/iss/build.h
  \brief Implements builders for IndSimpSet.
*/

#if !defined(__geom_mesh_iss_build_h__)
#define __geom_mesh_iss_build_h__

#include "stlib/geom/mesh/iss/IndSimpSetIncAdj.h"
#include "stlib/geom/mesh/iss/set.h"
#include "stlib/geom/mesh/iss/transform.h"

#include "stlib/geom/mesh/quadrilateral/QuadMesh.h"

#include "stlib/ads/iterator/IndirectIterator.h"

namespace stlib
{
namespace geom {

//-----------------------------------------------------------------------------
/*! \defgroup iss_build Builders
  These functions build indexed simplex sets from various inputs.
*/
//@{

//! Build from arrays of vertices and indexed simplices.
/*!
  \param vertices is the array of vertices.
  \param indexedSimplices is the array of indexed simplices.
*/
template<std::size_t SpaceD, std::size_t _M, typename _T>
inline
void
build(IndSimpSet<SpaceD, _M, _T>* mesh,
      const std::vector<typename IndSimpSet<SpaceD, _M, _T>::Vertex>& vertices,
      const std::vector<typename IndSimpSet<SpaceD, _M, _T>::IndexedSimplex>&
      indexedSimplices) {
   mesh->vertices = vertices;
   mesh->indexedSimplices = indexedSimplices;
   mesh->updateTopology();
}

//! Build from arrays of vertices, vertex identifiers and indexed simplices.
/*!
  \param vertices is the array of vertices.
  \param vertexIdentifiers is the array of vertex identifiers.
  \param simplices is the array of indexed simplices, which are
  tuples of the vertex identifiers.
*/
template<std::size_t SpaceD, std::size_t _M, typename _T>
inline
void
build(IndSimpSet<SpaceD, _M, _T>* mesh,
      const std::vector<typename IndSimpSet<SpaceD, _M, _T>::Vertex>& vertices,
      const std::vector<std::size_t>& vertexIdentifiers,
      const std::vector<typename IndSimpSet<SpaceD, _M, _T>::IndexedSimplex>&
      simplices) {
   // Start with simplices of vertex identifiers.
   mesh->vertices = vertices;
   mesh->indexedSimplices = simplices;
   mesh->convertFromIdentifiersToIndices(vertexIdentifiers);
   mesh->updateTopology();
}

//! Build from pointers to the vertices and indexed simplices.
/*!
  \param numVertices is the number of vertices.
  \param vertices points to the data for the vertices.
  \param numSimplices is the number of simplices.
  \param indexedSimplices points to the data for the indexed simplices.
*/
template<std::size_t SpaceD, std::size_t _M, typename _T>
inline
void
build(IndSimpSet<SpaceD, _M, _T>* mesh,
      const std::size_t numVertices,
      const typename IndSimpSet<SpaceD, _M, _T>::Vertex* vertices,
      const std::size_t numSimplices,
      const typename IndSimpSet<SpaceD, _M, _T>::IndexedSimplex* indexedSimplices) {
   std::vector<typename IndSimpSet<SpaceD, _M, _T>::Vertex>
      v(vertices, vertices + numVertices);
   mesh->vertices.swap(v);
   std::vector<typename IndSimpSet<SpaceD, _M, _T>::IndexedSimplex>
      is(indexedSimplices, indexedSimplices + numSimplices);
   mesh->indexedSimplices.swap(is);
   mesh->updateTopology();
}

//! Build from pointers to the vertices and indexed simplices.
/*!
  \param numVertices is the number of vertices.
  \param coordinates points to the data for the vertex coordinates.
  \param numSimplices is the number of simplices.
  \param indices points to the data for the simplex indices.
*/
template<std::size_t SpaceD, std::size_t _M, typename _T, typename _Index>
inline
void
build(IndSimpSet<SpaceD, _M, _T>* mesh,
      const std::size_t numVertices, const _T* coordinates,
      const std::size_t numSimplices, const _Index* indices) {
   mesh->vertices.resize(numVertices);
   mesh->indexedSimplices.resize(numSimplices);
   for (std::size_t i = 0; i != numVertices; ++i) {
      for (std::size_t n = 0; n != SpaceD; ++n) {
         mesh->vertices[i][n] = *coordinates++;
      }
   }
   for (std::size_t i = 0; i != numSimplices; ++i) {
      for (std::size_t m = 0; m != _M + 1; ++m) {
         mesh->indexedSimplices[i][m] = *indices++;
      }
   }
   mesh->updateTopology();
}

//! Build from the number of vertices and simplices.
/*!
  \param numVertices is the number of vertices.
  \param numSimplices is the number of simplices.
*/
template<std::size_t SpaceD, std::size_t _M, typename _T>
inline
void
build(IndSimpSet<SpaceD, _M, _T>* mesh,
      const std::size_t numVertices, const std::size_t numSimplices) {
   mesh->vertices.resize(numVertices);
   mesh->indexedSimplices.resize(numSimplices);
}

//! Build from the number of vertices and simplices.
/*!
  \param numVertices is the number of vertices.
  \param numSimplices is the number of simplices.
*/
template<std::size_t SpaceD, std::size_t _M, typename _T>
inline
void
build(IndSimpSetIncAdj<SpaceD, _M, _T>* mesh,
      const std::size_t numVertices, const std::size_t numSimplices) {
   build(static_cast<IndSimpSet<SpaceD, _M, _T>*>(mesh), numVertices, numSimplices);
   {
      // Clear the vertex-simplex incidence relations.
      typename IndSimpSetIncAdj<SpaceD, _M, _T>::IncidenceContainer empty;
      mesh->incident.swap(empty);
   }
   {
      // Clear the adjacency relations.
      typename IndSimpSetIncAdj<SpaceD, _M, _T>::AdjacencyContainer empty;
      mesh->adjacent.swap(empty);
   }
}


//! Build from a quadrilateral mesh.
/*!
  \relates IndSimpSet

  \param quadMesh The input quadrilateral mesh.
  \param mesh The output simplicial mesh.

  Each quadrilateral is split to form two triangles.  In the splitting,
  the shorter diagonal is chosen.
*/
template<std::size_t N, typename T>
void
buildFromQuadMesh(const QuadMesh<N, T>& quadMesh,
                  IndSimpSet<N, 2, T>* mesh);

//! Make a mesh from a subset of vertices of a mesh.
/*!
  \relates IndSimpSet

  \param in The input mesh.
  \param verticesBeginning The beginning of the range of vertex indices.
  \param verticesEnd The end of the range of vertex indices.
  \param out The output mesh.

  \c IntForIter is an integer forward iterator.
*/
template < std::size_t N, std::size_t M, typename T,
         typename IntForIter >
void
buildFromSubsetVertices(const IndSimpSet<N, M, T>& in,
                        IntForIter verticesBeginning,
                        IntForIter verticesEnd,
                        IndSimpSet<N, M, T>* out);


//! Make a new mesh from the subset of simplices.
/*!
  \relates IndSimpSet

  \param in is the input mesh.
  \param simplicesBeginning is the beginning of the range of simplex indices.
  \param simplicesEnd is the end of the range of simplex indices.
  \param out is the output mesh.

  \c IntForIter is an integer forward iterator.
*/
template < std::size_t N, std::size_t M, typename T,
         typename IntForIter >
void
buildFromSubsetSimplices(const IndSimpSet<N, M, T>& in,
                         IntForIter simplicesBeginning,
                         IntForIter simplicesEnd,
                         IndSimpSet<N, M, T>* out);


//! Make a mesh by selecting vertices from the input mesh that are inside the object.
/*!
  \relates IndSimpSet

  \param in is the input mesh.
  \param f is the level set function that describes the object.
  Points inside/outside the object have negative/positive values.
  \param out is the output mesh.

  \c LSF is the level set functor.
*/
template<std::size_t N, std::size_t M, typename T, class LSF>
void
buildFromVerticesInside(const IndSimpSet<N, M, T>& in,
                        const LSF& f,
                        IndSimpSet<N, M, T>* out);


//! Make a mesh by selecting simplices from the input mesh that are inside the object.
/*!
  \relates IndSimpSet

  \param in is the input mesh.
  \param f is the level set function that describes the object.
  Points inside/outside the object have negative/positive values.
  \param out is the output mesh.

  \c LSF is the level set functor.  A simplex is determined to be inside the
  object if its centroid is inside.
*/
template<std::size_t N, std::size_t M, typename T, class LSF>
void
buildFromSimplicesInside(const IndSimpSet<N, M, T>& in,
                         const LSF& f,
                         IndSimpSet<N, M, T>* out);


//! Make a mesh that is the boundary of the input mesh.
/*!
  \relates IndSimpSet

  \param in The input mesh.
  \param out The output mesh.
  \param usedVertexIndices The vertex indices that are used in the boundary.
*/
template < std::size_t N, std::size_t M, typename T,
         typename IntOutputIterator >
void
buildBoundary(const IndSimpSetIncAdj<N, M, T>& in,
              IndSimpSet < N, M - 1, T > * out,
              IntOutputIterator usedVertexIndices);


//! Make a mesh that is the boundary of the input mesh.
/*!
  \relates IndSimpSet

  \param in The input mesh.
  \param out The output mesh.
*/
template<std::size_t N, std::size_t M, typename T>
inline
void
buildBoundary(const IndSimpSetIncAdj<N, M, T>& in,
              IndSimpSet < N, M - 1, T > * out) {
   buildBoundary(in, out, ads::TrivialOutputIterator());
}


//! Make a mesh that is the boundary of the input mesh.
/*!
  \relates IndSimpSet

  This function does not pack the output mesh.  That is, it does not remove
  the unused interior vertices.

  \param in The input mesh.
  \param out The output mesh.
  \param incidentSimplices The incident simplex index of each boundary face
  is recorded in this output iterator.
*/
template < std::size_t N, std::size_t M, typename T,
         typename IntOutputIterator >
void
buildBoundaryWithoutPacking(const IndSimpSetIncAdj<N, M, T>& in,
                            IndSimpSet < N, M - 1, T > * out,
                            IntOutputIterator incidentSimplices);


//! Make a mesh that is the boundary of the input mesh.
/*!
  \relates IndSimpSet

  This function does not pack the output mesh.  That is, it does not remove
  the unused interior vertices.

  \param in The input mesh.
  \param out The output mesh.
*/
template<std::size_t N, std::size_t M, typename T>
inline
void
buildBoundaryWithoutPacking(const IndSimpSetIncAdj<N, M, T>& in,
                            IndSimpSet < N, M - 1, T > * out) {
   buildBoundaryWithoutPacking(in, out, ads::constructTrivialOutputIterator());
}



//! Make a mesh (separated into connected components) that is the boundary of the input mesh.
/*!
  \relates IndSimpSet

  This function does not pack the output mesh.  That is, it does not remove
  the unused interior vertices.

  \param in The input mesh.
  \param out The output mesh.
  \param delimiterIterator The \c delimiters define the components.
  Its values are the semi-open index ranges.
  \param incidentSimplices The incident simplex index of each boundary face
  is recorded in this output iterator.
*/
template < std::size_t N, std::size_t M, typename T,
         typename IntOutputIterator1, typename IntOutputIterator2 >
void
buildBoundaryOfComponentsWithoutPacking
(const IndSimpSetIncAdj<N, M, T>& in,
 IndSimpSet < N, M - 1, T > * out,
 IntOutputIterator1 delimiterIterator,
 IntOutputIterator2 incidentSimplices);



//! Make a mesh (separated into connected components) that is the boundary of the input mesh.
/*!
  \relates IndSimpSet

  This function does not pack the output mesh.  That is, it does not remove
  the unused interior vertices.

  \param in The input mesh.
  \param out The output mesh.
  \param delimiterIterator The \c delimiters define the components.
  Its values are the semi-open index ranges.
*/
template < std::size_t N, std::size_t M, typename T,
         typename IntOutputIterator >
inline
void
buildBoundaryOfComponentsWithoutPacking
(const IndSimpSetIncAdj<N, M, T>& in,
 IndSimpSet < N, M - 1, T > * out,
 IntOutputIterator delimiterIterator) {
   buildBoundaryOfComponentsWithoutPacking
   (in, out, delimiterIterator, ads::constructTrivialOutputIterator());
}



//! Make a mesh (separated into connected components) that is the boundary of the input mesh.
/*!
  \relates IndSimpSet

  \param in The input mesh.
  \param out The output mesh.
  \param delimiterIterator The \c delimiters define the components.
  Its values are the semi-open index ranges.
  \param incidentSimplices The incident simplex index of each boundary face
  is recorded in this output iterator.
*/
template < std::size_t N, std::size_t M, typename T,
         typename IntOutputIterator1, typename IntOutputIterator2 >
inline
void
buildBoundaryOfComponents(const IndSimpSetIncAdj<N, M, T>& in,
                          IndSimpSet < N, M - 1, T > * out,
                          IntOutputIterator1 delimiterIterator,
                          IntOutputIterator2 incidentSimplices) {
   // First do it without packing.
   buildBoundaryOfComponentsWithoutPacking(in, out, delimiterIterator,
                                           incidentSimplices);
   // Pack the mesh to get rid of the interior vertices.
   pack(out);
}




//! Make a mesh (separated into connected components) that is the boundary of the input mesh.
/*!
  \relates IndSimpSet

  \param in The input mesh.
  \param out The output mesh.
  \param delimiterIterator The \c delimiters define the components.
  Its values are the semi-open index ranges.
*/
template < std::size_t N, std::size_t M, typename T,
         typename IntOutputIterator >
inline
void
buildBoundaryOfComponents(const IndSimpSetIncAdj<N, M, T>& in,
                          IndSimpSet < N, M - 1, T > * out,
                          IntOutputIterator delimiterIterator) {
   buildBoundaryOfComponents(in, out, delimiterIterator,
                             ads::constructTrivialOutputIterator());
}




//! Make a mesh by connecting the boundary nodes to a new center point.
/*!
  \relates IndSimpSet

  \param boundary The input boundary mesh.
  \param mesh The output solid mesh.
*/
template<std::size_t N, std::size_t M, typename T>
void
centerPointMesh(const IndSimpSet < N, M - 1, T > & boundary,
                IndSimpSet<N, M, T>* mesh);


//! Merge a range of meshes to make a single mesh.
/*!
  \relates IndSimpSet

  \param beginning The beginning of a range of meshes.
  \param end The end of a range of meshes.
  \param out The output mesh.

  The meshes are simply concatenated.  Duplicate vertices (if any) are
  not removed.
*/
template < std::size_t N, std::size_t M, typename T,
         typename MeshInputIterator >
void
merge(MeshInputIterator beginning, MeshInputIterator end,
      IndSimpSet<N, M, T>* out);


//! Merge two meshes to make a single mesh.
/*!
  \relates IndSimpSet

  \param a The first input mesh.
  \param b The second input mesh.
  \param out The output mesh.

  The meshes are simply concatenated.  Duplicate vertices (if any) are
  not removed.
*/
template<std::size_t N, std::size_t M, typename T>
void
merge2(const IndSimpSet<N, M, T>& a, const IndSimpSet<N, M, T>& b,
       IndSimpSet<N, M, T>* out);

//@}

} // namespace geom
}

#define __geom_mesh_iss_build_ipp__
#include "stlib/geom/mesh/iss/build.ipp"
#undef __geom_mesh_iss_build_ipp__

#endif
