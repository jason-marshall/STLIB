// -*- C++ -*-

#if !defined(__geom_mesh_simplicial_manipulators_ipp__)
#error This file is an implementation detail.
#endif

namespace stlib
{
namespace geom {


template < std::size_t N, std::size_t M, typename T,
         template<class> class Node,
         template<class> class Cell,
         template<class, class> class Cont >
inline
void
orientPositive(SimpMeshRed<N, M, T, Node, Cell, Cont>* mesh) {
   typedef SimpMeshRed<N, M, T, Node, Cell, Cont> Mesh;
   typedef typename Mesh::Simplex Simplex;
   typedef typename Mesh::CellIterator CellIterator;

   // This only makes sense if the simplex dimension is the same as
   // the space dimension.
   static_assert(N == M, "The space and simplex dimension must be the same.");

   Simplex s;
   SimplexJac<N, T> sj;
   // For each cell.
   for (CellIterator i = mesh->getCellsBeginning(); i != mesh->getCellsEnd();
         ++i) {
      mesh->getSimplex(i, &s);
      sj.setFunction(s);
      // If the content is negative.
      if (sj.getDeterminant() < 0) {
         // Reverse its orientation.
         i->reverseOrientation();
      }
   }
}



//! Erase the nodes that do not have an incident cells.
template < std::size_t N, std::size_t M, typename T,
         template<class> class Node,
         template<class> class Cell,
         template<class, class> class Cont >
inline
void
eraseUnusedNodes(SimpMeshRed<N, M, T, Node, Cell, Cont>* mesh) {
   typedef SimpMeshRed<N, M, T, Node, Cell, Cont> SMR;
   typedef typename SMR::NodeIterator NodeIterator;

   // For each node.
   for (NodeIterator i = mesh->getNodesBeginning(); i != mesh->getNodesEnd();) {
      if (i->getCellsSize() == 0) {
         mesh->eraseNode(i++);
      }
      else {
         ++i;
      }
   }
}



//! Remove cells until there are none with minimum adjacencies less than specified.
template < std::size_t N, std::size_t M, typename T,
         template<class> class Node,
         template<class> class Cell,
         template<class, class> class Cont >
inline
void
eraseCellsWithLowAdjacencies(SimpMeshRed<N, M, T, Node, Cell, Cont>* mesh,
                             const std::size_t minAdjacencies) {
   typedef SimpMeshRed<N, M, T, Node, Cell, Cont> SMR;
   typedef typename SMR::CellIterator CellIterator;

   // Make sure that the min adjacency requirement is in the right range.
   assert(minAdjacencies <= M + 1);

   // Loop until the cells with low adjacencies are gone.
   std::size_t count;
   do {
      count = 0;
      // Loop over the cells.
      for (CellIterator i = mesh->getCellsBeginning();
            i != mesh->getCellsEnd();) {
         // If this cell has a low number of adjacencies.
         if (i->getNumberOfNeighbors() < minAdjacencies) {
            mesh->eraseCell(i++);
            ++count;
         }
         else {
            ++i;
         }
      }
   }
   while (count != 0);

   eraseUnusedNodes(mesh);
}



// Re-number the node and cell identifiers so they start at 0 and are
// contiguous.
template < std::size_t N, std::size_t M, typename T,
         template<class> class Node,
         template<class> class Cell,
         template<class, class> class Cont >
void
renumberIdentifiers(SimpMeshRed<N, M, T, Node, Cell, Cont>* mesh) {
   typedef SimpMeshRed<N, M, T, Node, Cell, Cont> Mesh;
   typedef typename Mesh::NodeIterator NodeIterator;
   typedef typename Mesh::CellIterator CellIterator;

   // Set the node identifiers.
   std::size_t index = 0;
   for (NodeIterator i = mesh->getNodesBeginning(); i != mesh->getNodesEnd();
         ++i, ++index) {
      i->setIdentifier(index);
   }

   // Set the cell identifiers.
   index = 0;
   for (CellIterator i = mesh->getCellsBeginning(); i != mesh->getCellsEnd();
         ++i, ++index) {
      i->setIdentifier(index);
   }
}

} // namespace geom
}
