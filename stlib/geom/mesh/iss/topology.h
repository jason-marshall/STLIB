// -*- C++ -*-

/*!
  \file geom/mesh/iss/topology.h
  \brief Topological functions for indexed simplex sets.
*/

#if !defined(__geom_mesh_iss_topology_h__)
#define __geom_mesh_iss_topology_h__

#include "stlib/geom/mesh/iss/IndSimpSetIncAdj.h"

namespace stlib
{
namespace geom {

//-----------------------------------------------------------------------------
/*! \defgroup iss_topology Topology Functions for IndSimpSet
*/
//@{

//! Count the connected components of the mesh.
/*! \relates IndSimpSetIncAdj */
template<std::size_t N, std::size_t M, typename T>
std::size_t
countComponents(const IndSimpSetIncAdj<N, M, T>& mesh);

//! Return true if the simplices share a vertex.
/*! \relates IndSimpSetIncAdj */
template<std::size_t N, std::size_t M, typename T>
bool
doSimplicesShareAnyVertex(const IndSimpSetIncAdj<N, M, T>& mesh,
                          const std::size_t i, const std::size_t j);

//@}

} // namespace geom
}

#define __geom_mesh_iss_topology_ipp__
#include "stlib/geom/mesh/iss/topology.ipp"
#undef __geom_mesh_iss_topology_ipp__

#endif
