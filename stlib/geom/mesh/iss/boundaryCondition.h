// -*- C++ -*-

/*!
  \file geom/mesh/iss/boundaryCondition.h
  \brief Boundary condition functions for indexed simplex sets.
*/

#if !defined(__geom_mesh_iss_boundaryCondition_h__)
#define __geom_mesh_iss_boundaryCondition_h__

#include "stlib/geom/mesh/iss/geometry.h"
#include "stlib/geom/mesh/iss/ISS_SignedDistance.h"

namespace stlib
{
namespace geom {

//-----------------------------------------------------------------------------
/*! \defgroup iss_boundaryCondition Boundary condition functions for simplicial meshes. */
//@{

//! Apply the closest point boundary condition at a vertex.
template<typename T, class ISS>
void
applyBoundaryCondition(IndSimpSetIncAdj<2, 1, T>* mesh,
                       const ISS_SD_ClosestPoint<ISS>& condition,
                       std::size_t n);

//! Apply the closest point in the normal direction boundary condition at a vertex.
template < typename T,
         class ISS >
void
applyBoundaryCondition(IndSimpSetIncAdj<2, 1, T>* mesh,
                       const ISS_SD_ClosestPointDirection<ISS>& condition,
                       std::size_t n);

//! Apply the closest point boundary condition at a vertex.
template<typename T, class ISS>
void
applyBoundaryCondition(IndSimpSetIncAdj<3, 2, T>* mesh,
                       const ISS_SD_ClosestPoint<ISS>& condition,
                       std::size_t n);

//! Apply the closest point in the normal direction boundary condition at a vertex.
template < typename T,
         class ISS >
void
applyBoundaryCondition(IndSimpSetIncAdj<3, 2, T>* mesh,
                       const ISS_SD_ClosestPointDirection<ISS>& condition,
                       std::size_t n);

//! Apply the condition at a vertex.
/*!
  \note The vertex may be in the interior or on the boundary.
*/
template < std::size_t N, typename T,
         class UnaryFunction >
void
applyBoundaryCondition(IndSimpSetIncAdj<N, N, T>* mesh,
                       const UnaryFunction& condition,
                       std::size_t n);

//! Apply the closest point boundary condition at a vertex.
template<std::size_t N, typename T, class ISS>
void
applyBoundaryCondition(IndSimpSetIncAdj<N, N, T>* mesh,
                       const ISS_SD_ClosestPoint<ISS>& condition,
                       std::size_t n);

//! Apply the closer point boundary condition at a vertex.
template<std::size_t N, typename T, class ISS>
void
applyBoundaryCondition(IndSimpSetIncAdj<N, N, T>* mesh,
                       const ISS_SD_CloserPoint<ISS>& condition,
                       std::size_t n);

//! Apply the closest point in the normal direction boundary condition at a vertex.
template<std::size_t N, typename T, class ISS>
void
applyBoundaryCondition(IndSimpSetIncAdj<N, N, T>* mesh,
                       const ISS_SD_ClosestPointDirection<ISS>& condition,
                       std::size_t n);

//! Apply the closer point in the normal direction boundary condition at a vertex.
template<std::size_t N, typename T, class ISS>
void
applyBoundaryCondition(IndSimpSetIncAdj<N, N, T>* mesh,
                       const ISS_SD_CloserPointDirection<ISS>& condition,
                       std::size_t n);

//@}

} // namespace geom
}

#define __geom_mesh_iss_boundaryCondition_ipp__
#include "stlib/geom/mesh/iss/boundaryCondition.ipp"
#undef __geom_mesh_iss_boundaryCondition_ipp__

#endif
