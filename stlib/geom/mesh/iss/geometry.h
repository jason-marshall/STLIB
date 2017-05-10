// -*- C++ -*-

/*!
  \file geom/mesh/iss/geometry.h
  \brief Geometric functions for indexed simplex sets.
*/

#if !defined(__geom_mesh_iss_geometry_h__)
#define __geom_mesh_iss_geometry_h__

#include "stlib/geom/mesh/iss/IndSimpSetIncAdj.h"

#include "stlib/geom/mesh/simplex/geometry.h"

namespace stlib
{
namespace geom {

//-----------------------------------------------------------------------------
/*! \defgroup iss_geometry Geometric functions for simplicial meshes. */
//@{

//
// Normal
//

//! Return the outward normal at the specified vertex.
template<typename T>
typename IndSimpSetIncAdj<2, 1, T>::Vertex
computeVertexNormal(const IndSimpSetIncAdj<2, 1, T>& mesh, std::size_t n);


//! Compute the outward normal at the specified vertex.
template<typename T>
void
computeVertexNormal(const IndSimpSetIncAdj<2, 1, T>& mesh, std::size_t n,
                    typename IndSimpSetIncAdj<2, 1, T>::Vertex* normal);


//! Return the outward normal at the specified boundary vertex.
template<typename T>
typename IndSimpSetIncAdj<2, 2, T>::Vertex
computeVertexNormal(const IndSimpSetIncAdj<2, 2, T>& mesh, std::size_t n);


//! Compute the outward normal at the specified boundary vertex.
template<typename T>
void
computeVertexNormal(const IndSimpSetIncAdj<2, 2, T>& mesh, std::size_t n,
                    typename IndSimpSetIncAdj<2, 2, T>::Vertex* normal);


//! Return the outward normal at the specified vertex.
template<typename T>
typename IndSimpSetIncAdj<3, 2, T>::Vertex
computeVertexNormal(const IndSimpSetIncAdj<3, 2, T>& mesh, std::size_t n);


//! Compute the outward normal at the specified vertex.
template<typename T>
void
computeVertexNormal(const IndSimpSetIncAdj<3, 2, T>& mesh, std::size_t n,
                    typename IndSimpSetIncAdj<3, 2, T>::Vertex* normal);


//! Return the outward normal at the specified boundary vertex.
template<typename T>
typename IndSimpSetIncAdj<3, 3, T>::Vertex
computeVertexNormal(const IndSimpSetIncAdj<3, 3, T>& mesh, std::size_t n);


//! Compute the outward normal at the specified boundary vertex.
template<typename T>
void
computeVertexNormal(const IndSimpSetIncAdj<3, 3, T>& mesh, std::size_t n,
                    typename IndSimpSetIncAdj<3, 3, T>::Vertex* normal);



//! Compute the outward normal for the specified simplex (triangle face).
template<typename T>
void
computeSimplexNormal(const IndSimpSetIncAdj<3, 2, T>& mesh,
                     std::size_t simplexIndex,
                     typename IndSimpSetIncAdj<3, 2, T>::Vertex* simplexNormal);


//! Compute the outward normals for the simplices (triangle faces).
template<typename T>
void
computeSimplexNormals(const IndSimpSetIncAdj<3, 2, T>& mesh,
                      std::vector<typename IndSimpSetIncAdj<3, 2, T>::Vertex>*
                      simplexNormals);


//! Compute the outward normals for the simplices (line segments).
template<typename T>
void
computeSimplexNormals(const IndSimpSetIncAdj<2, 1, T>& mesh,
                      std::vector<typename IndSimpSetIncAdj<2, 1, T>::Vertex>*
                      simplexNormals);


//! Compute the outward normals for the vertices.
template<typename T>
void
computeVertexNormals
(const IndSimpSetIncAdj<3, 2, T>& mesh,
 const std::vector<typename IndSimpSetIncAdj<3, 2, T>::Vertex>& simplexNormals,
 std::vector<typename IndSimpSetIncAdj<3, 2, T>::Vertex>* vertexNormals);


//! Compute the outward normals for the simplices and vertices.
template<typename T>
void
computeSimplexAndVertexNormals
(const IndSimpSetIncAdj<3, 2, T>& mesh,
 std::vector<typename IndSimpSetIncAdj<3, 2, T>::Vertex>* simplexNormals,
 std::vector<typename IndSimpSetIncAdj<3, 2, T>::Vertex>* vertexNormals);



//! Return the cosine of the interior angle at the specified vertex.
/*!
  \pre The vertex must have two incident simplices.
 */
template<std::size_t N, typename T>
T
computeCosineAngle(const IndSimpSetIncAdj<N, 1, T>& mesh,
                   std::size_t vertexIndex);


//! Return the cosine of the interior angle at the specified 1-face.
/*!
  \pre The 1-face must have two incident simplices.
 */
template<typename T>
T
computeCosineAngle(const IndSimpSetIncAdj<3, 2, T>& mesh,
                   const typename IndSimpSetIncAdj<3, 2, T>::Face& face);


//! Return the cosine of the interior angle at the specified boundary vertex.
/*!
  \pre It must be a boundary vertex.

  The angle is in the range [0..pi].
*/
template<typename T>
T
computeCosineBoundaryAngle(const IndSimpSetIncAdj<3, 2, T>& mesh,
                           std::size_t vertexIndex);


//! Return the solid interior angle at the specified vertex.
template<typename T>
T
computeAngle(const IndSimpSetIncAdj<3, 2, T>& mesh, std::size_t n);


//! Return the sum of the incident angles at the specified vertex.
template<typename T>
T
computeAngle(const IndSimpSetIncAdj<3, 3, T>& mesh, std::size_t n);

//! Return the sum of the incident angles at the specified vertex.
template<typename T>
T
computeAngle(const IndSimpSetIncAdj<2, 2, T>& mesh, std::size_t n);

//! Return the sum of the incident angles at the specified vertex.
template<typename T>
T
computeAngle(const IndSimpSetIncAdj<1, 1, T>& mesh, std::size_t n);


//! Project the line segments to 1-D and collect them.
template<typename T, typename OutputIterator>
void
projectAndGetSimplices(const IndSimpSet<2, 1, T>& mesh,
                       OutputIterator simplices);


//! Project the triangle simplices to 2-D and collect them.
template<typename T, typename OutputIterator>
void
projectAndGetSimplices(const IndSimpSet<3, 2, T>& mesh,
                       OutputIterator simplices);

//@}

} // namespace geom
}

#define __geom_mesh_iss_geometry_ipp__
#include "stlib/geom/mesh/iss/geometry.ipp"
#undef __geom_mesh_iss_geometry_ipp__

#endif
