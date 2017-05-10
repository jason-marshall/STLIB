// -*- C++ -*-

#if !defined(__geom_mesh_simplicial_build_ipp__)
#error This file is an implementation detail.
#endif

namespace stlib
{
namespace geom {


//! Make an IndSimpSet from a SimpMeshRed.
/*!
  \c ISSV is the Indexed Simplex Set Vertex type.
  \c ISSIS is the Indexed Simplex Set Indexed Simplex type.
*/
template < std::size_t N, std::size_t M, typename T,
         template<class> class Node,
         template<class> class Cell,
         template<class, class> class Cont >
void
buildIndSimpSetFromSimpMeshRed(const SimpMeshRed<N, M, T, Node, Cell, Cont>& smr,
                               IndSimpSet<N, M, T>* iss) {
   typedef IndSimpSet<N, M, T> ISS;
   typedef typename ISS::VertexIterator VertexIterator;
   typedef typename ISS::IndexedSimplexIterator IndexedSimplexIterator;

   typedef SimpMeshRed<N, M, T, Node, Cell, Cont> SMR;
   typedef typename SMR::NodeConstIterator NodeConstIterator;
   typedef typename SMR::CellConstIterator CellConstIterator;

   // Set the vertices.
   iss->vertices.resize(smr.computeNodesSize());
   VertexIterator vertexIterator = iss->vertices.begin();
   for (NodeConstIterator i = smr.getNodesBeginning(); i != smr.getNodesEnd();
         ++i, ++vertexIterator) {
      *vertexIterator = i->getVertex();
   }

   // Set the indexed simplices.
   smr.setNodeIdentifiers();
   iss->indexedSimplices.resize(smr.computeCellsSize());
   IndexedSimplexIterator simplexIterator = iss->indexedSimplices.begin();
   for (CellConstIterator i = smr.getCellsBeginning(); i != smr.getCellsEnd();
         ++i, ++simplexIterator) {
      for (std::size_t m = 0; m != M + 1; ++m) {
         (*simplexIterator)[m] = i->getNode(m)->getIdentifier();
      }
   }

   // Update the topology.
   iss->updateTopology();
}

} // namespace geom
}
