// -*- C++ -*-

/*!
  \file geom/mesh/iss/accessors.h
  \brief Implements accessors for IndSimpSet.
*/

#if !defined(__geom_mesh_iss_accessors_h__)
#define __geom_mesh_iss_accessors_h__

#include "stlib/geom/mesh/iss/IndSimpSetIncAdj.h"

namespace stlib
{
namespace geom {

//-----------------------------------------------------------------------------
/*! \defgroup iss_accessors Accessors */
//@{


//! Get the previous vertex index.
/*! \relates IndSimpSetIncAdj */
template<std::size_t N, typename T>
int
getPreviousVertexIndex(const IndSimpSetIncAdj<N, 1, T>& mesh, std::size_t n);


//! Get the next vertex index.
/*! \relates IndSimpSetIncAdj */
template<std::size_t N, typename T>
int
getNextVertexIndex(const IndSimpSetIncAdj<N, 1, T>& mesh, std::size_t n);


//! Get the previous vertex.
/*! \relates IndSimpSetIncAdj */
template<std::size_t N, typename T>
inline
typename IndSimpSetIncAdj<N, 1, T>::Vertex
getPreviousVertex(const IndSimpSetIncAdj<N, 1, T>& mesh, const std::size_t n) {
   return mesh.vertices[getPreviousVertexIndex(mesh, n)];
}


//! Get the next vertex.
/*! \relates IndSimpSetIncAdj */
template<std::size_t N, typename T>
inline
typename IndSimpSetIncAdj<N, 1, T>::Vertex
getNextVertex(const IndSimpSetIncAdj<N, 1, T>& mesh, const std::size_t n) {
   return mesh.vertices[getNextVertexIndex(mesh, n)];
}



//! Get the face in the simplex that is opposite to the given vertex.
/*! \relates IndSimpSet */
template<std::size_t N, std::size_t M, typename T>
void
getFace(const IndSimpSet<N, M, T>& mesh,
        int simplexIndex, std::size_t vertexIndex,
        typename IndSimpSet<N, M, T>::SimplexFace* face);


//! Get the indexed face in the simplex that is opposite to the given vertex.
/*! \relates IndSimpSet */
template<std::size_t N, std::size_t M, typename T>
void
getIndexedFace(const IndSimpSet<N, M, T>& mesh,
               std::size_t simplexIndex, std::size_t vertexIndex,
               typename IndSimpSet<N, M, T>::IndexedSimplexFace* face);


//! Get the centroid of the specified simplex.
/*! \relates IndSimpSet */
template<std::size_t N, std::size_t M, typename T>
void
getCentroid(const IndSimpSet<N, M, T>& mesh, std::size_t n,
            typename IndSimpSet<N, M, T>::Vertex* x);


//! Return true if the simplices of the mesh have consistent orientations.
/*! \relates IndSimpSetIncAdj */
template<std::size_t N, std::size_t M, typename T>
bool
isOriented(const IndSimpSetIncAdj<N, M, T>& mesh);


//@}

} // namespace geom
}

#define __geom_mesh_iss_accessors_ipp__
#include "stlib/geom/mesh/iss/accessors.ipp"
#undef __geom_mesh_iss_accessors_ipp__

#endif
