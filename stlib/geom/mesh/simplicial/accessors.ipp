// -*- C++ -*-

#if !defined(__geom_mesh_simplicial_accessors_ipp__)
#error This file is an implementation detail.
#endif

namespace stlib
{
namespace geom {


// Get the incident cells of the edge.
template<typename SMR, typename CellIteratorOutputIterator>
inline
void
getIncidentCells(const typename SMR::cell_iterator cell,
                 const std::size_t i, const std::size_t j,
                 CellIteratorOutputIterator out) {
   typedef typename SMR::Node Node;
   typedef typename Node::CellIteratorIterator CellIteratorIterator;

   // The simplex dimension must be 3.
   static_assert(SMR::M == 3, "The simplex dimension must be 3.");

   typename SMR::Node* const a = cell->getNode(i);
   typename SMR::Node* const b = cell->getNode(j);

   for (CellIteratorIterator c = a->getCellIteratorsBeginning();
         c != a->getCellIteratorsEnd(); ++c) {
      // If the cell is incident to the other node as well, it is incident
      // to the edge.
      if ((*c)->hasNode(b)) {
         // Record the cell iterator.
         *out++ = *c;
      }
   }
}



// For a 2-simplex cell, a pair of nodes defines a 1-face.
// Return the index of this 1-face.
template<class CellIterator, class NodeIterator>
inline
std::size_t
getFaceIndex(const CellIterator& cell,
             const NodeIterator& a, const NodeIterator& b) {
   // This function may only be used with 2-simplices.
   typedef typename std::iterator_traits<CellIterator>::value_type CellType;
   static_assert(CellType::M == 2, "The simplex dimension must be 2.");

   const std::size_t i = cell->getIndex(a);
   const std::size_t j = cell->getIndex(b);
   std::size_t k = 0;
   if (k == i || k == j) {
      ++k;
   }
   if (k == i || k == j) {
      ++k;
   }
   return k;
}



template < std::size_t N, std::size_t M, typename T,
         template<class> class Node,
         template<class> class Cell,
         template<class, class> class Cont >
inline
bool
isOriented(const SimpMeshRed<N, M, T, Node, Cell, Cont>& mesh) {
   typedef SimpMeshRed<N, M, T, Node, Cell, Cont> SMR;
   typedef typename SMR::CellConstIterator CellConstIterator;
   typedef std::array < std::size_t, M + 1 > IndexedSimplex;
   typedef std::array<std::size_t, M> IndexedFace;

   std::size_t i, m, mu;
   IndexedSimplex s, t;
   IndexedFace f, g;
   CellConstIterator d;

   // For each cell.
   for (CellConstIterator c = mesh.cells_begin(); c != mesh.cells_end(); ++c) {
      // Get the indexed simplex for c.
      for (i = 0; i != M + 1; ++i) {
         s[i] = c->getNode(i)->getIdentifier();
      }
      // For each adjacent cell.
      for (m = 0; m != M + 1; ++m) {
         // The m_th adjacent cell.
         d = c->getNeighbor(m);
         // If this is not a boundary face.
         if (d != 0) {
            // Get the indexed simplex for d.
            for (i = 0; i != M + 1; ++i) {
               t[i] = d->getNode(i)->getIdentifier();
            }
            mu = c->getMirrorIndex(m);
            s.getFace(m, f);
            t.getFace(mu, g);
            g.reverseOrientation();
            if (! haveSameOrientation(f, g)) {
               /* CONTINUE REMOVE
               std::cout << s << "    " << t << "\n"
                     << f << "    " << g << "\n";
               */
               return false;
            }
         }
      }
   }

   return true;
}

} // namespace geom
}
