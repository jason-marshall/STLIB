// -*- C++ -*-

#if !defined(__geom_mesh_simplicial_transform_ipp__)
#error This file is an implementation detail.
#endif

namespace stlib
{
namespace geom {

// Transform each vertex in the range with the specified function.
template<typename SMR, typename NodeIterInIter, class UnaryFunction>
inline
void
transformNodes(NodeIterInIter begin, NodeIterInIter end,
               const UnaryFunction& f) {
   typedef typename SMR::Node Node;
   Node* i;
   for (; begin != end; ++begin) {
      i = *begin;
      i->setVertex(f(i->getVertex()));
   }
}


// Transform each vertex in the mesh with the specified function.
template < std::size_t N, std::size_t M, typename T,
         template<class> class Node,
         template<class> class Cell,
         template<class, class> class Cont,
         class UnaryFunction >
inline
void
transform(SimpMeshRed<N, M, T, Node, Cell, Cont>* mesh, const UnaryFunction& f) {
   typedef typename SimpMeshRed<N, M, T, Node, Cell, Cont>::NodeIterator
   NodeIterator;
   for (NodeIterator i = mesh->getNodesBeginning(); i != mesh->getNodesEnd();
         ++i) {
      i->setVertex(f(i->getVertex()));
   }
}


// Transform each boundary vertex in the mesh with the specified function.
template < std::size_t N, std::size_t M, typename T,
         template<class> class Node,
         template<class> class Cell,
         template<class, class> class Cont,
         class UnaryFunction >
inline
void
transformBoundary(SimpMeshRed<N, M, T, Node, Cell, Cont>* mesh,
                  const UnaryFunction& f) {
   typedef typename SimpMeshRed<N, M, T, Node, Cell, Cont>::NodeIterator
   NodeIterator;
   for (NodeIterator i = mesh->getNodesBeginning(); i != mesh->getNodesEnd();
         ++i) {
      if (i->isOnBoundary()) {
         i->setVertex(f(i->getVertex()));
      }
   }
}

} // namespace geom
}
