// -*- C++ -*-

/*!
  \file geom/mesh/simplicial/build.h
  \brief Functions to build edges in a SimpMeshRed<2,2>.
*/

#if !defined(__geom_mesh_simplicial_build_h__)
#define __geom_mesh_simplicial_build_h__

#include "stlib/geom/mesh/simplicial/SimpMeshRed.h"

namespace stlib
{
namespace geom {

//! Make an indexed simplex set from the mesh.
/*!
  \c ISSV is the Indexed Simplex Set Vertex type.
  \c ISSIS is the Indexed Simplex Set Indexed Simplex type.
*/
template < std::size_t N, std::size_t M, typename T,
         template<class> class Node,
         template<class> class Cell,
         template<class, class> class Cont >
void
buildIndSimpSetFromSimpMeshRed(const SimpMeshRed<N, M, T, Node, Cell, Cont>& smr,
                               IndSimpSet<N, M, T>* iss);

} // namespace geom
}

#define __geom_mesh_simplicial_build_ipp__
#include "stlib/geom/mesh/simplicial/build.ipp"
#undef __geom_mesh_simplicial_build_ipp__

#endif
