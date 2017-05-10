// -*- C++ -*-

/*!
  \file geom/mesh/simplicial/insert.h
  \brief Insert cells into a SimpMeshRed.
*/

#if !defined(__geom_mesh_simplicial_insert_h__)
#define __geom_mesh_simplicial_insert_h__

#include "stlib/geom/mesh/simplicial/SimpMeshRed.h"

namespace stlib
{
namespace geom {

//! Insert the specified cell into the 1-D mesh.
template < std::size_t N, typename T,
         template<class> class _Node,
         template<class> class _Cell,
         template<class, class> class Cont >
inline
typename SimpMeshRed<N, 1, T, _Node, _Cell, Cont>::CellIterator
insertCell
(SimpMeshRed<N, 1, T, _Node, _Cell, Cont>* mesh,
 typename SimpMeshRed<N, 1, T, _Node, _Cell, Cont>::Node* n0,
 typename SimpMeshRed<N, 1, T, _Node, _Cell, Cont>::Node* n1,
 typename SimpMeshRed<N, 1, T, _Node, _Cell, Cont>::Cell* c0 = 0,
 typename SimpMeshRed<N, 1, T, _Node, _Cell, Cont>::Cell* c1 = 0) {
   typedef typename SimpMeshRed<N, 1, T, _Node, _Cell, Cont>::Cell Cell;
   return mesh->insertCell(Cell(n0, n1, c0, c1));
}

//! Insert the specified cell into the 2-D mesh.
template < std::size_t N, typename T,
         template<class> class _Node,
         template<class> class _Cell,
         template<class, class> class Cont >
inline
typename SimpMeshRed<N, 2, T, _Node, _Cell, Cont>::CellIterator
insertCell
(SimpMeshRed<N, 2, T, _Node, _Cell, Cont>* mesh,
 typename SimpMeshRed<N, 2, T, _Node, _Cell, Cont>::Node* n0,
 typename SimpMeshRed<N, 2, T, _Node, _Cell, Cont>::Node* n1,
 typename SimpMeshRed<N, 2, T, _Node, _Cell, Cont>::Node* n2,
 typename SimpMeshRed<N, 2, T, _Node, _Cell, Cont>::Cell* c0 = 0,
 typename SimpMeshRed<N, 2, T, _Node, _Cell, Cont>::Cell* c1 = 0,
 typename SimpMeshRed<N, 2, T, _Node, _Cell, Cont>::Cell* c2 = 0) {
   typedef typename SimpMeshRed<N, 2, T, _Node, _Cell, Cont>::Cell Cell;
   return mesh->insertCell(Cell(n0, n1, n2, c0, c1, c2));
}

//! Insert the specified cell into the 3-D mesh.
template < std::size_t N, typename T,
         template<class> class _Node,
         template<class> class _Cell,
         template<class, class> class Cont >
inline
typename SimpMeshRed<N, 3, T, _Node, _Cell, Cont>::CellIterator
insertCell
(SimpMeshRed<N, 3, T, _Node, _Cell, Cont>* mesh,
 typename SimpMeshRed<N, 3, T, _Node, _Cell, Cont>::Node* n0,
 typename SimpMeshRed<N, 3, T, _Node, _Cell, Cont>::Node* n1,
 typename SimpMeshRed<N, 3, T, _Node, _Cell, Cont>::Node* n2,
 typename SimpMeshRed<N, 3, T, _Node, _Cell, Cont>::Node* n3,
 typename SimpMeshRed<N, 3, T, _Node, _Cell, Cont>::Cell* c0 = 0,
 typename SimpMeshRed<N, 3, T, _Node, _Cell, Cont>::Cell* c1 = 0,
 typename SimpMeshRed<N, 3, T, _Node, _Cell, Cont>::Cell* c2 = 0,
 typename SimpMeshRed<N, 3, T, _Node, _Cell, Cont>::Cell* c3 = 0) {
   typedef typename SimpMeshRed<N, 3, T, _Node, _Cell, Cont>::Cell Cell;
   return mesh->insertCell(Cell(n0, n1, n2, n3, c0, c1, c2, c3));
}

} // namespace geom
}

#endif
