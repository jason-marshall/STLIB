// -*- C++ -*-

#if !defined(__geom_mesh_simplicial_quality_ipp__)
#error This file is an implementation detail.
#endif

namespace stlib
{
namespace geom {


//! Calculate the adjacency counts for the simplices in the mesh.
template < std::size_t N, std::size_t M, typename T,
         template<class> class Node,
         template<class> class Cell,
         template<class, class> class Cont >
inline
void
countAdjacencies(const SimpMeshRed<N, M, T, Node, Cell, Cont>& mesh,
                 std::array < std::size_t, M + 2 > * counts) {
   typedef SimpMeshRed<N, M, T, Node, Cell, Cont> SMR;
   typedef typename SMR::CellConstIterator CellConstIterator;

   std::fill(counts->begin(), counts->end(), 0);
   // For each cell.
   CellConstIterator i = mesh.getCellsBeginning();
   const CellConstIterator iEnd = mesh.getCellsEnd();
   for (; i != iEnd; ++i) {
      ++(*counts)[i->getNumberOfNeighbors()];
   }
}



//! Calculate edge length statistics.
template < std::size_t N, std::size_t M, typename T,
         template<class> class Node,
         template<class> class Cell,
         template<class, class> class Cont >
inline
void
computeEdgeLengthStatistics(const SimpMeshRed<N, M, T, Node, Cell, Cont>& mesh,
                            T* minLength, T* maxLength, T* meanLength) {
   typedef SimpMeshRed<N, M, T, Node, Cell, Cont> SMR;
   typedef typename SMR::EdgeConstIterator EdgeConstIterator;

   *minLength = std::numeric_limits<T>::max();
   *maxLength = 0;
   *meanLength = 0;

   std::size_t num = 0;
   T d;
   for (EdgeConstIterator i = mesh.getEdgesBeginning();
         i != mesh.getEdgesEnd(); ++i, ++num) {
      d = ext::euclideanDistance(i->first->getNode(i->second)->getVertex(),
                                 i->first->getNode(i->third)->getVertex());
      if (d < *minLength) {
         *minLength = d;
      }
      if (d > *maxLength) {
         *maxLength = d;
      }
      *meanLength += d;
   }

   if (num != 0) {
      *meanLength /= num;
   }
}

} // namespace geom
}
