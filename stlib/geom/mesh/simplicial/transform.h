// -*- C++ -*-

/*!
  \file geom/mesh/simplicial/transform.h
  \brief Implements operations that transform a SimpMeshRed.
*/

#if !defined(__geom_mesh_simplicial_transform_h__)
#define __geom_mesh_simplicial_transform_h__

#include "stlib/geom/mesh/simplicial/SimpMeshRed.h"

namespace stlib
{
namespace geom {


//! Transform each vertex in the range with the specified function.
/*!
  The first template argument must be specified explicitly.
*/
template<typename SMR, typename NodeIterInIter, class UnaryFunction>
void
transformNodes(NodeIterInIter begin, NodeIterInIter end,
               const UnaryFunction& f);


//! Transform each vertex in the mesh with the specified function.
template < std::size_t N, std::size_t M, typename T,
         template<class> class Node,
         template<class> class Cell,
         template<class, class> class Cont,
         class UnaryFunction >
void
transform(SimpMeshRed<N, M, T, Node, Cell, Cont>* mesh, const UnaryFunction& f);


//! Transform each boundary vertex in the mesh with the specified function.
template < std::size_t N, std::size_t M, typename T,
         template<class> class Node,
         template<class> class Cell,
         template<class, class> class Cont,
         class UnaryFunction >
void
transformBoundary(SimpMeshRed<N, M, T, Node, Cell, Cont>* mesh,
                  const UnaryFunction& f);


} // namespace geom
}

#define __geom_mesh_simplicial_transform_ipp__
#include "stlib/geom/mesh/simplicial/transform.ipp"
#undef __geom_mesh_simplicial_transform_ipp__

#endif
