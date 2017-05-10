// -*- C++ -*-

/*!
  \file geom/mesh/simplicial/valid.h
  \brief Functions to valid edges in a SimpMeshRed<2,2>.
*/

#if !defined(__geom_mesh_simplicial_valid_h__)
#define __geom_mesh_simplicial_valid_h__

#include "stlib/geom/mesh/simplicial/SimpMeshRed.h"

#include "stlib/geom/mesh/simplex/SimplexJac.h"

#include "stlib/ads/algorithm/unique.h"

namespace stlib
{
namespace geom {

//! Return true if the mesh is valid.
template < std::size_t N, std::size_t M, typename T,
         template<class> class Node,
         template<class> class Cell,
         template<class, class> class Cont >
bool
isValid(const SimpMeshRed<N, M, T, Node, Cell, Cont>& mesh);

} // namespace geom
}

#define __geom_mesh_simplicial_valid_ipp__
#include "stlib/geom/mesh/simplicial/valid.ipp"
#undef __geom_mesh_simplicial_valid_ipp__

#endif
