// -*- C++ -*-

#if !defined(__geom_mesh_simplicial_set_ipp__)
#error This file is an implementation detail.
#endif

namespace stlib
{
namespace geom {

//! Get the nodes that are outside the object.
template<typename NodeInIter, class LSF, typename OutIter>
inline
void
determineNodesOutside(NodeInIter begin, NodeInIter end,
                      const LSF& f, OutIter iter) {
   // Loop over the nodes.
   for (; begin != end; ++begin) {
      // If the vertex is outside the object.
      if (f(begin->getVertex()) > 0) {
         // Insert it into the container.
         *iter++ = begin;
      }
   }
}




//! Get the cells whose centroids are outside the object.
template<typename CellInIter, class LSF, typename OutIter>
inline
void
determineCellsOutside(CellInIter begin, CellInIter end,
                      const LSF& f, OutIter iter) {
   typedef typename std::iterator_traits<CellInIter>::value_type Cell;
   typedef typename Cell::Vertex Vertex;

   Vertex x;
   // Loop over the cells.
   for (; begin != end; ++begin) {
      begin->getCentroid(&x);
      // If the centroid is outside the object.
      if (f(x) > 0) {
         // Append the cell iterator into the container.
         *iter++ = begin;
      }
   }
}




//! Get the node iterators for the mesh.
template<typename NodeInIter, typename OutIter>
inline
void
getNodes(NodeInIter begin, NodeInIter end, OutIter iter) {
   // For each node.
   for (; begin != end; ++begin) {
      // Append the node.
      *iter++ = &*begin;
   }
}




//! Get the node iterators for the interior nodes.
template<typename NodeInIter, typename OutIter>
inline
void
determineInteriorNodes(NodeInIter begin, NodeInIter end, OutIter iter) {
   // For each node.
   for (; begin != end; ++begin) {
      // If this is an interior node.
      if (! begin->isOnBoundary()) {
         // Append the node.
         *iter++ = begin;
      }
   }
}




//! Get the node iterators for the boundary nodes.
template<typename NodeInIter, typename OutIter>
inline
void
determineBoundaryNodes(NodeInIter begin, NodeInIter end, OutIter iter) {
   // For each node.
   for (; begin != end; ++begin) {
      // If this is an boundary node.
      if (begin->isOnBoundary()) {
         // Append the node.
         *iter++ = &*begin;
      }
   }
}




//! Get the cell iterators with at least the specified number of adjacencies.
template<typename CellInIter, typename OutIter>
inline
void
determineCellsWithRequiredAdjacencies(CellInIter begin, CellInIter end,
                                      const std::size_t minimumRequiredAdjacencies,
                                      OutIter iter) {
   // For each cell.
   for (; begin != end; ++begin) {
      // If this cell has the minimum required number of adjacencies.
      if (begin->getNumberOfNeighbors() >= minimumRequiredAdjacencies) {
         // Append the cell.
         *iter++ = begin;
      }
   }
}




//! Get the cell iterators with adjacencies less than specified.
template<typename CellInIter, typename OutIter>
inline
void
determineCellsWithLowAdjacencies(CellInIter begin, CellInIter end,
                                 const std::size_t minimumRequiredAdjacencies,
                                 OutIter iter) {
   // For each cell.
   for (; begin != end; ++begin) {
      // If this cell has a low number of adjacencies.
      if (begin->getNumberOfNeighbors() < minimumRequiredAdjacencies) {
         // Append the cell.
         *iter++ = begin;
      }
   }
}



// Get the neighboring nodes of a node.
/*
  The set of nodes (not including the specified node) that share a cell with
  the specified node.
*/
template<std::size_t N, std::size_t _M, typename _T,
         template<class> class _Node,
         template<class> class _Cell,
         template<class, class> class _Cont>
inline
void
determineNeighbors(SimpMeshRed<N, _M, _T, _Node, _Cell, _Cont>& /*mesh*/,
                   typename SimpMeshRed<N, _M, _T, _Node, _Cell, _Cont>::
                   Node* const node,
                   typename SimpMeshRed<N, _M, _T, _Node, _Cell, _Cont>::
                   NodePointerSet* neighbors) {
   typedef SimpMeshRed<N, _M, _T, _Node, _Cell, _Cont> SMR;
   typedef typename SMR::Node Node;
   typedef typename SMR::Node::CellIncidentToNodeIterator
   CellIncidentToNodeIterator;

   neighbors->clear();

   CellIncidentToNodeIterator c;
   Node* n;
   // For each incident cell.
   for (c = node->getCellsBeginning(); c != node->getCellsEnd(); ++c) {
      // For each node of the cell.
      for (std::size_t m = 0; m != _M + 1; ++m) {
         // The node.
         n = c->getNode(m);
         if (n != node) {
            // Add it to the set.
            neighbors->insert(n);
         }
      }
   }
}



// Get the neighboring boundary nodes of a node.
/*
  The set of boundary nodes (not including the specified node) that share
  a boundary face with the specified node.
*/
template<typename SMR>
inline
void
determineBoundaryNeighbors(typename SMR::Node* node,
                           typename SMR::NodePointerSet* neighbors) {
   typedef typename SMR::Node Node;
   typedef typename SMR::Node::CellIncidentToNodeIterator
   CellIncidentToNodeIterator;

   const std::size_t M = 2;

   neighbors->clear();

   CellIncidentToNodeIterator c;
   Node* n;
   // For each incident cell.
   for (c = node->getCellsBeginning(); c != node->getCellsEnd(); ++c) {
      // For each node of the cell.
      for (std::size_t m = 0; m != M + 1; ++m) {
         // The node.
         n = c->getNode(m);
         // The three tests are arranged from least to most expensive.
         if (n != node && neighbors->count(n) == 0 &&
               c->isFaceOnBoundary(getFaceIndex(c, node, n))) {
            // Add it to the set.
            neighbors->insert(n);
         }
      }
   }
}



//! Get all the nodes within the specified radius of the specified node.
/*!
  The set includes the specified node.
*/
template<std::size_t N, std::size_t _M, typename _T,
         template<class> class _Node,
         template<class> class _Cell,
         template<class, class> class _Cont>
inline
void
determineNeighbors(SimpMeshRed<N, _M, _T, _Node, _Cell, _Cont>& /*mesh*/,
                   typename SimpMeshRed<N, _M, _T, _Node, _Cell, _Cont>::
                   Node* node,
                   std::size_t radius,
                   typename SimpMeshRed<N, _M, _T, _Node, _Cell, _Cont>::
                   NodePointerSet* neighbors) {
   typedef SimpMeshRed<N, _M, _T, _Node, _Cell, _Cont> SMR;
   typedef typename SMR::Node Node;
   typedef typename SMR::NodePointerSet NodePointerSet;
   typedef typename SMR::Node::CellIncidentToNodeIterator
   CellIncidentToNodeIterator;

   neighbors->clear();
   // The zero radius neighbors is the node itself.
   neighbors->insert(node);

   NodePointerSet front;
   CellIncidentToNodeIterator c;
   Node* n;
   while (radius--) {
      // For each node in the set.
      for (typename NodePointerSet::const_iterator i = neighbors->begin();
            i != neighbors->end(); ++i) {
         n = *i;
         // For each incident cell.
         for (c = n->getCellsBeginning(); c != n->getCellsEnd(); ++c) {
            // For each node of the cell.
            for (std::size_t m = 0; m != _M + 1; ++m) {
               // Add the node to the front.
               front.insert(c->getNode(m));
            }
         }
      }
      neighbors->insert(front.begin(), front.end());
      front.clear();
   }
}




//! Get the faces of the incident cells.
template<std::size_t N, std::size_t _M, typename _T,
         template<class> class _Node,
         template<class> class _Cell,
         template<class, class> class _Cont>
inline
void
determineFacesOfIncidentCells
(SimpMeshRed<N, _M, _T, _Node, _Cell, _Cont>& /*mesh*/,
 typename SimpMeshRed<N, _M, _T, _Node, _Cell, _Cont>::Node* node,
 typename SimpMeshRed<N, _M, _T, _Node, _Cell, _Cont>::FaceSet* faces) {
   typedef SimpMeshRed<N, _M, _T, _Node, _Cell, _Cont> SMR;
   typedef typename SMR::Node::CellIncidentToNodeIterator
   CellIncidentToNodeIterator;
   typedef typename SMR::Face Face;

   Face face;
   std::size_t m;

   // Loop over the incident cells.
   for (CellIncidentToNodeIterator i = node->getCellsBeginning();
         i != node->getCellsEnd(); ++i) {
      face.first = *i.base();
      // Loop over faces of the cells.
      for (m = 0; m != _M + 1; ++m) {
         face.second = m;
         faces->insert(face);
      }
   }
}



// Build a set of cell iterators from a range of cell identifiers.
template < std::size_t N, std::size_t M, typename T,
         template<class> class Node,
         template<class> class Cell,
         template<class, class> class Cont,
         typename IntInIter >
inline
void
convertIdentifiersToIterators(SimpMeshRed<N, M, T, Node, Cell, Cont>& mesh,
                              IntInIter begin, IntInIter end,
                              typename SimpMeshRed<N, M, T, Node, Cell, Cont>::
                              CellIteratorSet* cells) {
   typedef SimpMeshRed<N, M, T, Node, Cell, Cont> SMR;
   typedef typename SMR::CellIterator CellIterator;

   // Make a vector of the cell identifiers.
   std::vector<std::size_t> identifiers;
   for (; begin != end; ++begin) {
      identifiers.push_back(*begin);
   }

   // The identifiers must be in sorted order.
   if (! std::is_sorted(identifiers.begin(), identifiers.end())) {
      std::sort(identifiers.begin(), identifiers.end());
   }

#ifdef STLIB_DEBUG
   //
   // Check the identifier values.
   //

   // Determine the maximum identifier.
   std::size_t maximumIdentifier = 0;
   {
      CellIterator c = mesh.getCellsEnd();
      --c;
      if (c != mesh.getCellsEnd()) {
         maximumIdentifier = c->getIdentifier();
      }
   }
   if (identifiers.size() != 0) {
      assert(identifiers[identifiers.size() - 1] <= maximumIdentifier);
   }
#endif

   // Make a set of the cell identifiers.
   cells->clear();
   CellIterator c = mesh.getCellsBeginning();
   std::vector<std::size_t>::const_iterator i = identifiers.begin();
   for (; c != mesh.getCellsEnd() && i != identifiers.end(); ++c) {
      if (c->getIdentifier() == *i) {
         // Since the cell identifiers are in sorted order, give it the hint
         // to insert it at the end.
         cells->insert(cells->end(), c);
         ++i;
      }
   }
}

} // namespace geom
}
