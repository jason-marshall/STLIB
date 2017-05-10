// -*- C++ -*-

/*!
  \file geom/mesh/iss/distance.h
  \brief Compute the distance to a simplicial mesh for a single point.
*/

#if !defined(__geom_mesh_iss_distance_h__)
#define __geom_mesh_iss_distance_h__

#include "stlib/geom/mesh/iss/geometry.h"

#include "stlib/geom/mesh/simplex/simplex_distance.h"

#include "stlib/ads/algorithm/min_max.h"
#include "stlib/ads/iterator/TrivialOutputIterator.h"

namespace stlib
{
namespace geom {

//-----------------------------------------------------------------------------
/*! \defgroup iss_distance Compute the distance to a simplicial mesh for a single point. */
//@{

//---------------------------------------------------------------------------
// Signed distance, 2-D space, 1-manifold, single point.
//---------------------------------------------------------------------------

//! Compute the signed distance to the mesh and closest point on the mesh.
/*!
  Use this function to compute the distance to a mesh for a single point.
  Or call it a few times if you want to compute the signed distance for a
  small number of points.  This function has linear computational complexity
  in the size of the mesh.  If you are computing the distance for many points,
  use \c ISS_SignedDistance instead.

  This function will return correct results only if the distance is
  well-defined.  The mesh must be a 1-manifold.  If the mesh has a boundary,
  the closest point must lie on the interior.  If the closest point is
  on the boundary, the signed distance is not defined.  (The unsigned
  distance, however, is defined.)

  The algorithm for computing the signed distance first uses the
  vertices of the mesh to obtain an upper bound on the squared
  distance to the mesh.  Then the signed distance is computed to those
  vertices and faces which could possibly contain the closest point.
*/
template<typename _T>
typename IndSimpSetIncAdj<2, 1, _T>::Number
computeSignedDistance
(const IndSimpSetIncAdj<2, 1, _T>& mesh,
 const std::vector<typename IndSimpSetIncAdj<2, 1, _T>::Number>&
 squaredHalfLengths,
 const typename IndSimpSetIncAdj<2, 1, _T>::Vertex& point,
 typename IndSimpSetIncAdj<2, 1, _T>::Vertex* closestPoint);

//! Compute the signed distance to the mesh and closest point on the mesh.
/*!
  This function computes the squared half lengths of the faces and then
  calls the above function.
*/
template<typename _T>
typename IndSimpSetIncAdj<2, 1, _T>::Number
computeSignedDistance
(const IndSimpSetIncAdj<2, 1, _T>& mesh,
 const typename IndSimpSetIncAdj<2, 1, _T>::Vertex& point,
 typename IndSimpSetIncAdj<2, 1, _T>::Vertex* closestPoint);

//! Compute the signed distance to the mesh.
/*!
  This function just calls the above function which computes the signed
  distance and closest point.
*/
template<typename _T>
inline
typename IndSimpSetIncAdj<2, 1, _T>::Number
computeSignedDistance
(const IndSimpSetIncAdj<2, 1, _T>& mesh,
 const typename IndSimpSetIncAdj<2, 1, _T>::Vertex& point) {
   typename IndSimpSetIncAdj<2, 1, _T>::Vertex closestPoint;
   return computeSignedDistance(mesh, point, &closestPoint);
}

//---------------------------------------------------------------------------
// Signed distance, 2-D space, 1-manifold, multiple points.
//---------------------------------------------------------------------------

//! Compute the signed distances to the mesh and closest points on the mesh.
template < typename _T,
         typename InputIterator, typename NumberOutputIterator,
         typename PointOutputIterator >
void
computeSignedDistance
(const IndSimpSetIncAdj<2, 1, _T>& mesh,
 InputIterator pointsBeginning, InputIterator pointsEnd,
 NumberOutputIterator distances, PointOutputIterator closestPoints);

//! Compute the signed distances to the mesh and closest points on the mesh.
template < typename _T,
         typename InputIterator, typename NumberOutputIterator >
inline
void
computeSignedDistance
(const IndSimpSetIncAdj<2, 1, _T>& mesh,
 InputIterator pointsBeginning, InputIterator pointsEnd,
 NumberOutputIterator distances) {
   computeSignedDistance(mesh, pointsBeginning, pointsEnd, distances,
                         ads::constructTrivialOutputIterator());
}

//---------------------------------------------------------------------------
// Signed distance, 3-D space, 2-manifold, single point.
//---------------------------------------------------------------------------

//! Compute the signed distance to the mesh and closest point on the mesh.
/*!
  Use this function to compute the distance to a mesh for a single point.
  Or call it a few times if you want to compute the signed distance for a
  small number of points.  This function has linear computational complexity
  in the size of the mesh.  If you are computing the distance for many points,
  use \c ISS_SignedDistance instead.

  This function will return correct results only if the distance is
  well-defined.  The mesh must be a 2-manifold.  If the mesh has a boundary,
  the closest point must lie on the interior.  If the closest point is
  on the boundary, the signed distance is not defined.  (The unsigned
  distance, however, is defined.)

  The algorithm for computing the signed distance first uses the
  vertices of the mesh to obtain an upper bound on the squared
  distance to the mesh.  Then the signed distance is computed to those
  vertices, edges, and faces which could possibly contain the closest point.
*/
template<typename _T>
typename IndSimpSetIncAdj<3, 2, _T>::Number
computeSignedDistance
(const IndSimpSetIncAdj<3, 2, _T>& mesh,
 const std::vector<typename IndSimpSetIncAdj<3, 2, _T>::Number>&
 squaredLongestEdgeLengths,
 const typename IndSimpSetIncAdj<3, 2, _T>::Vertex& point,
 typename IndSimpSetIncAdj<3, 2, _T>::Vertex* closestPoint);

//! Compute the signed distance to the mesh and closest point on the mesh.
/*!
  This function computes the squared edge lengths and then calls the
  above function.
*/
template<typename _T>
typename IndSimpSetIncAdj<3, 2, _T>::Number
computeSignedDistance
(const IndSimpSetIncAdj<3, 2, _T>& mesh,
 const typename IndSimpSetIncAdj<3, 2, _T>::Vertex& point,
 typename IndSimpSetIncAdj<3, 2, _T>::Vertex* closestPoint);

//! Compute the signed distance to the mesh.
/*!
  This function just calls the above function which computes the signed
  distance and closest point.
*/
template<typename _T>
inline
typename IndSimpSetIncAdj<3, 2, _T>::Number
computeSignedDistance
(const IndSimpSetIncAdj<3, 2, _T>& mesh,
 const typename IndSimpSetIncAdj<3, 2, _T>::Vertex& point) {
   typename IndSimpSetIncAdj<3, 2, _T>::Vertex closestPoint;
   return computeSignedDistance(mesh, point, &closestPoint);
}

//---------------------------------------------------------------------------
// Signed distance, 3-D space, 2-manifold, multiple points.
//---------------------------------------------------------------------------

//! Compute the signed distances to the mesh and closest points on the mesh.
template < typename _T,
         typename InputIterator, typename NumberOutputIterator,
         typename PointOutputIterator >
void
computeSignedDistance
(const IndSimpSetIncAdj<3, 2, _T>& mesh,
 InputIterator pointsBeginning, InputIterator pointsEnd,
 NumberOutputIterator distances, PointOutputIterator closestPoints);

//! Compute the signed distances to the mesh and closest points on the mesh.
template < typename _T,
         typename InputIterator, typename NumberOutputIterator >
inline
void
computeSignedDistance
(const IndSimpSetIncAdj<3, 2, _T>& mesh,
 InputIterator pointsBeginning, InputIterator pointsEnd,
 NumberOutputIterator distances) {
   computeSignedDistance(mesh, pointsBeginning, pointsEnd, distances,
                         ads::constructTrivialOutputIterator());
}

//@}

} // namespace geom
}

#define __geom_mesh_iss_distance_ipp__
#include "stlib/geom/mesh/iss/distance.ipp"
#undef __geom_mesh_iss_distance_ipp__

#endif
