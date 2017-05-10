// -*- C++ -*-

#if !defined(__geom_StructuredGrid_ipp__)
#error This file is an implementation detail of the class StructuredGrid.
#endif

namespace stlib
{
namespace geom {

//
// File I/O member functions.
//

// Forward declaration.
template<std::size_t N, std::size_t M, typename T>
void
structuredGridWriteIndexedSimplexSet(std::ostream& out,
                                     const StructuredGrid<N, M, T>& x);


// Implementation for 2-D grids.
template<std::size_t N, typename T>
inline
void
structuredGridWriteIndexedSimplexSet(std::ostream& out,
                                     const StructuredGrid<N, 2, T>& sg) {
   typedef typename StructuredGrid<N, 2, T>::ConstIterator ConstIterator;
   typedef typename StructuredGrid<N, 2, T>::Array Array;
   typedef typename Array::Index Index;
   typedef typename Array::IndexList IndexList;
   // The number of triangles is twice the number of quadrilaterals.
   const std::size_t numTriangles = 2 * (sg.getExtents()[0] - 1) *
                                    (sg.getExtents()[1] - 1);
   // Write the number of nodes and the number of triangles.
   out << sg.getSize() << " " << numTriangles << '\n';
   // Write the nodes.
   for (ConstIterator i = sg.getBeginning(); i != sg.getEnd(); ++i) {
      out << *i << '\n';
   }
   // Write the element indices.
   const Index iEnd = sg.getExtents()[0] - 1;
   const Index jEnd = sg.getExtents()[1] - 1;
   for (Index i = 0; i != iEnd; ++i) {
      for (Index j = 0; j != jEnd; ++j) {
         // Choose the shorter diagonal of the quadrilateral.
        if (ext::squaredDistance(sg(IndexList{{i, j}}),
                                 sg(IndexList{{i + 1, j + 1}})) <
            ext::squaredDistance(sg(IndexList{{i + 1, j}}),
                                 sg(IndexList{{i, j + 1}}))) {
            std::cout
                  << sg.getGrid().arrayIndex(IndexList{{i, j}}) << " "
                  << sg.getGrid().arrayIndex(IndexList{{i + 1, j}}) << " "
                  << sg.getGrid().arrayIndex(IndexList{{i + 1, j + 1}}) << '\n'
                  << sg.getGrid().arrayIndex(IndexList{{i + 1, j + 1}}) << " "
                  << sg.getGrid().arrayIndex(IndexList{{i, j + 1}}) << " "
                  << sg.getGrid().arrayIndex(IndexList{{i, j}}) << '\n';
         }
         else {
            std::cout
                  << sg.getGrid().arrayIndex(IndexList{{i, j}}) << " "
                  << sg.getGrid().arrayIndex(IndexList{{i + 1, j}}) << " "
                  << sg.getGrid().arrayIndex(IndexList{{i, j + 1}}) << '\n'
                  << sg.getGrid().arrayIndex(IndexList{{i + 1, j + 1}}) << " "
                  << sg.getGrid().arrayIndex(IndexList{{i, j + 1}}) << " "
                  << sg.getGrid().arrayIndex(IndexList{{i + 1, j}}) << '\n';
         }
      }
   }
}


template<std::size_t N, std::size_t M, typename T>
inline
void
StructuredGrid<N, M, T>::
writeIndexedSimplexSet(std::ostream& out) const {
   structuredGridWriteIndexedSimplexSet(out, *this);
}

} // namespace geom
}
