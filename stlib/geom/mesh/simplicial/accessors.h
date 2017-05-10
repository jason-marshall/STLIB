// -*- C++ -*-

/*!
  \file geom/mesh/simplicial/accessors.h
  \brief Implements accessors operations for SimpMeshRed.
*/

#if !defined(__geom_mesh_simplicial_accessors_h__)
#define __geom_mesh_simplicial_accessors_h__

#include "stlib/geom/mesh/simplicial/build.h"

#include "stlib/geom/mesh/iss/accessors.h"

namespace stlib
{
namespace geom {


//! Get the incident cells of the edge.
template<typename SMR, typename CellIteratorOutputIterator>
void
getIncidentCells(const typename SMR::CellIterator cell,
                 std::size_t i, std::size_t j,
                 CellIteratorOutputIterator out);


//! For a 2-simplex cell, a pair of nodes defines a 1-face.  Return the index of this 1-face.
template<class CellIterator, class NodeIterator>
std::size_t
getFaceIndex(const CellIterator& cell,
             const NodeIterator& a, const NodeIterator& b);


//! Return true if the simplices of the mesh have consistent orientations.
/*! \relates SimpMeshRed */
template < std::size_t N, std::size_t M, typename T,
         template<class> class Node,
         template<class> class Cell,
         template<class, class> class Cont >
bool
isOriented(const SimpMeshRed<N, M, T, Node, Cell, Cont>& mesh);


} // namespace geom
}

#define __geom_mesh_simplicial_accessors_ipp__
#include "stlib/geom/mesh/simplicial/accessors.ipp"
#undef __geom_mesh_simplicial_accessors_ipp__

#endif
