// -*- C++ -*-

#if !defined(__geom_mesh_simplicial_SmrCell_ipp__)
#error This file is an implementation detail of the class SmrCell.
#endif

namespace stlib
{
namespace geom {


// Return true if the cell has a boundary face incident to the edge.
template<class SMR>
inline
bool
doesCellHaveIncidentFaceOnBoundary(const typename SMR::CellConstIterator& c,
                                   const std::size_t i, const std::size_t j) {
   static_assert(SMR::M == 3, "The simplex dimension must be 3.");

   // For each face.
   for (std::size_t m = 0; m != SMR::M + 1; ++m) {
      // The i_th and j_th faces are not incident to the edge.
      // Select the incident faces.
      if (m != i && m != j) {
         // If the incident face is on the boundary.
         //if (c->getNeighbor(m) == 0) {
         if (c->isFaceOnBoundary(m)) {
            return true;
         }
      }
   }
   // No incident boundary faces.
   return false;
}


// Return true if the edge is on the boundary.
// \c i and \c j are the indices of the edge in the cell.
// An edge is on the boundary iff an incident face is on the boundary.
template<class SMR>
inline
bool
isOnBoundary(const typename SMR::CellConstIterator& c,
             const std::size_t i, const std::size_t j) {
   typedef typename SMR::Node Node;
   typedef typename Node::CellIteratorConstIterator CellIteratorIterator;

   static_assert(SMR::M == 3, "The simplex dimension must be 3.");

   const Node* a = c->getNode(i);
   const Node* b = c->getNode(j);

   // For each cell incident to the source node.
   for (CellIteratorIterator c = a->getCellIteratorsBeginning();
         c != a->getCellIteratorsEnd(); ++c) {
      // If the cell is incident to the target node as well, it is incident
      // to the edge.
      if ((*c)->hasNode(b)) {
         // If there is an incident face on the boundary.
         if (doesCellHaveIncidentFaceOnBoundary<SMR>(*c, (*c)->getIndex(a),
               (*c)->getIndex(b))) {
            // The edge is on the boundary as well.
            return true;
         }
      }
   }

   // If there are no incident faces on the boundary, the edge is not on
   // the boundary.
   return false;
}


} // namespace geom
}
