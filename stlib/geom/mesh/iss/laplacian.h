// -*- C++ -*-

/*!
  \file geom/mesh/iss/laplacian.h
  \brief Implements Laplacian smoothing.
*/

#if !defined(__geom_mesh_iss_laplacian_h__)
#define __geom_mesh_iss_laplacian_h__

#include "stlib/geom/mesh/iss/IndSimpSetIncAdj.h"
#include "stlib/geom/mesh/iss/PointsOnManifold.h"
#include "stlib/geom/mesh/iss/accessors.h"
#include "stlib/geom/mesh/iss/geometry.h"

#include "stlib/numerical/constants.h"
#include "stlib/container/PackedArrayOfArrays.h"

#include <set>

namespace stlib
{
namespace geom {

//-----------------------------------------------------------------------------
/*! \defgroup iss_laplacian Laplacian Smoothing
*/
//@{


//! Perform sweeps of Laplacian smoothing on the interior vertices.
/*!
  \relates IndSimpSetIncAdj

  \param mesh Pointer to the simplicial mesh.
  \param numSweeps The number of smoothing sweeps.  By default it is one.
*/
template<std::size_t N, std::size_t M, typename T>
void
applyLaplacian(IndSimpSetIncAdj<N, M, T>* mesh, std::size_t numSweeps = 1);


//! Perform sweeps of Laplacian smoothing on the vertices.
/*!
  \relates IndSimpSetIncAdj

  \param mesh Pointer to the simplicial mesh.
  \param condition The functor that returns the closest point on the boundary.
  \param maxAngleDeviation Used to define corner features.  Nodes that are
  corner features will not be moved.
  \param numSweeps The number of smoothing sweeps.  By default it is one.

  Perform Laplacian smoothing on a 2-1 mesh (a line segment mesh in 2-D).
*/
template<typename T, class BoundaryCondition>
void
applyLaplacian(IndSimpSetIncAdj<2, 1, T>* mesh,
               const BoundaryCondition& condition,
               T maxAngleDeviation, std::size_t numSweeps = 1);


//! Perform sweeps of Laplacian smoothing on the vertices.
/*!
  \relates IndSimpSetIncAdj

  \param mesh Pointer to the simplicial mesh.
  \param manifold The boundary manifold data structure.
  \param numSweeps The number of smoothing sweeps.  By default it is one.
*/
template<typename T, std::size_t SD>
void
applyLaplacian(IndSimpSetIncAdj<2, 1, T>* mesh,
               PointsOnManifold<2, 1, SD, T>* manifold,
               std::size_t numSweeps = 1);


//! Perform sweeps of Laplacian smoothing on the boundary vertices.
/*!
  \relates IndSimpSetIncAdj

  \param mesh Pointer to the simplicial mesh.
  \param condition The functor that returns the closest point on the boundary.
  \param maxAngleDeviation Used to define corner features.  Nodes that are
  corner features will not be moved.
  \param numSweeps The number of smoothing sweeps.  By default it is one.
*/
template<typename T, class BoundaryCondition>
void
applyLaplacian(IndSimpSetIncAdj<3, 2, T>* mesh,
               const BoundaryCondition& condition,
               T maxAngleDeviation, std::size_t numSweeps = 1);


//@}

} // namespace geom
}

#define __geom_mesh_iss_laplacian_ipp__
#include "stlib/geom/mesh/iss/laplacian.ipp"
#undef __geom_mesh_iss_laplacian_ipp__

#endif
