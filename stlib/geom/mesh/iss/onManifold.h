// -*- C++ -*-

/*!
  \file geom/mesh/iss/onManifold.h
  \brief Implements on-manifold operations for IndSimpSet.
*/

#if !defined(__geom_mesh_iss_onManifold_h__)
#define __geom_mesh_iss_onManifold_h__

#include "stlib/geom/mesh/iss/IndSimpSetIncAdj.h"
#include "stlib/geom/mesh/iss/ISS_SimplexQuery.h"

#include "stlib/ads/iterator/IntIterator.h"

namespace stlib
{
namespace geom {

//-----------------------------------------------------------------------------
/*! \defgroup iss_onManifold Vertices on a manifold
  These functions determine the vertices on a manifold.
*/
//@{

//! Get the vertices on the manifold.
/*! \relates IndSimpSet */
template < std::size_t N, std::size_t M, typename T,
         std::size_t MM, typename MT, typename IntOutIter >
void
determineVerticesOnManifold(const IndSimpSet<N, M, T>& mesh,
                            const IndSimpSet<N, MM, MT>& manifold,
                            IntOutIter indexIterator,
                            T epsilon =
                               std::sqrt(std::numeric_limits<T>::epsilon()));


//! Get the vertices (from the set of vertices) on the manifold.
/*! \relates IndSimpSet */
template < std::size_t N, std::size_t M, typename T,
         typename IntInIter,
         std::size_t MM, typename MT, typename IntOutIter >
void
determineVerticesOnManifold(const IndSimpSet<N, M, T>& mesh,
                            IntInIter indicesBeginning, IntInIter indicesEnd,
                            const IndSimpSet<N, MM, MT>& manifold,
                            IntOutIter indexIterator,
                            T epsilon =
                               std::sqrt(std::numeric_limits<T>::epsilon()));

//@}

} // namespace geom
}

#define __geom_mesh_iss_onManifold_ipp__
#include "stlib/geom/mesh/iss/onManifold.ipp"
#undef __geom_mesh_iss_onManifold_ipp__

#endif
