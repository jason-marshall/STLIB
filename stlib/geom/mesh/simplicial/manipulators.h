// -*- C++ -*-

/*!
  \file geom/mesh/simplicial/manipulators.h
  \brief Implements operations that manipulators a SimpMeshRed.
*/

#if !defined(__geom_mesh_simplicial_manipulators_h__)
#define __geom_mesh_simplicial_manipulators_h__

#include "stlib/geom/mesh/simplicial/SimpMeshRed.h"

#include "stlib/geom/mesh/simplex/SimplexJac.h"

namespace stlib
{
namespace geom {


//! Orient each simplex so it has non-negative volume.
template < std::size_t N, std::size_t M, typename T,
         template<class> class Node,
         template<class> class Cell,
         template<class, class> class Cont >
void
orientPositive(SimpMeshRed<N, M, T, Node, Cell, Cont>* x);


//! Erase the nodes that do not have an incident cells.
template < std::size_t N, std::size_t M, typename T,
         template<class> class Node,
         template<class> class Cell,
         template<class, class> class Cont >
void
eraseUnusedNodes(SimpMeshRed<N, M, T, Node, Cell, Cont>* x);


//! Erase cells until there are none with minimum adjacencies less than specified.
template < std::size_t N, std::size_t M, typename T,
         template<class> class Node,
         template<class> class Cell,
         template<class, class> class Cont >
void
eraseCellsWithLowAdjacencies(SimpMeshRed<N, M, T, Node, Cell, Cont>* mesh,
                             std::size_t minimumAdjacencies);


//! Re-number the node and cell identifiers so they start at 0 and are contiguous.
template < std::size_t N, std::size_t M, typename T,
         template<class> class Node,
         template<class> class Cell,
         template<class, class> class Cont >
void
renumberIdentifiers(SimpMeshRed<N, M, T, Node, Cell, Cont>* x);


} // namespace geom
}

#define __geom_mesh_simplicial_manipulators_ipp__
#include "stlib/geom/mesh/simplicial/manipulators.ipp"
#undef __geom_mesh_simplicial_manipulators_ipp__

#endif
