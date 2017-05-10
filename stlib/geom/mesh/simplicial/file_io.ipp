// -*- C++ -*-

#if !defined(__geom_mesh_simplicial_file_io_ipp__)
#error This file is an implementation detail.
#endif

namespace stlib
{
namespace geom {


//! Write a mesh as an indexed simplex set in ascii format.
template < std::size_t N, std::size_t M, typename T,
         template<class> class Node,
         template<class> class Cell,
         template<class, class> class Cont >
inline
void
writeAscii(std::ostream& out, const SimpMeshRed<N, M, T, Node, Cell, Cont>& x) {
   // Set the vertex identifiers.
   x.setNodeIdentifiers();

   writeIssAscii<N, M>(out, x.getVerticesBeginning(), x.getVerticesEnd(),
                       x.getIndexedSimplicesBeginning(),
                       x.getIndexedSimplicesEnd());
}


//! Print detailed information about the mesh.
template < std::size_t N, std::size_t M, typename T,
         template<class> class Node,
         template<class> class Cell,
         template<class, class> class Cont >
inline
void
print(std::ostream& out, const SimpMeshRed<N, M, T, Node, Cell, Cont>& x) {
   typedef SimpMeshRed<N, M, T, Node, Cell, Cont> SMR;
   typedef typename SMR::NodeConstIterator NodeConstIterator;
   typedef typename SMR::CellConstIterator CellConstIterator;

   // Set the vertex identifiers and cell identifiers.
   x.setNodeIdentifiers();
   x.setCellIdentifiers();

   // Write the nodes.
   out << "Nodes:\n";
   for (NodeConstIterator i = x.getNodesBeginning(); i != x.getNodesEnd();
         ++i) {
      i->put(out);
   }
   // Write the indexed cells.
   out << "Cells:\n";
   for (CellConstIterator i = x.getCellsBeginning(); i != x.getCellsEnd();
         ++i) {
      i->put(out);
   }
}


//! Write a mesh as an indexed simplex set in binary format.
template < std::size_t N, std::size_t M, typename T,
         template<class> class Node,
         template<class> class Cell,
         template<class, class> class Cont >
void
writeBinary(std::ostream& out, const SimpMeshRed<N, M, T, Node, Cell, Cont>& x) {
   // Set the vertex identifiers.
   x.setNodeIdentifiers();

   writeIssBinary<N, M>(out, x.getVerticesBeginning(), x.getVerticesEnd(),
                        x.getIndexedSimplicesBeginning(),
                        x.getIndexedSimplicesEnd());
}


//! Read a mesh as an indexed simplex set in ascii format.
template < std::size_t N, std::size_t M, typename T,
         template<class> class Node,
         template<class> class Cell,
         template<class, class> class Cont >
inline
void
readAscii(std::istream& in, SimpMeshRed<N, M, T, Node, Cell, Cont>* x) {
   // Read an indexed simplex set.
   IndSimpSet<N, M, T> iss;
   readAscii(in, &iss);
   // Build the mesh from the indexed simplex set.
   x->build(iss);
}


//! Read an indexed simplex set in binary format.
template < std::size_t N, std::size_t M, typename T,
         template<class> class Node,
         template<class> class Cell,
         template<class, class> class Cont >
inline
void
readBinary(std::istream& in, SimpMeshRed<N, M, T, Node, Cell, Cont>* x) {
   // Read an indexed simplex set.
   IndSimpSet<N, M, T> iss;
   readBinary(in, &iss);
   // Build the mesh from the indexed simplex set.
   x->build(iss);
}


//! Write in VTK XML unstructured grid format.
template < std::size_t N, std::size_t M, typename T,
         template<class> class Node,
         template<class> class Cell,
         template<class, class> class Cont >
inline
void
writeVtkXml(std::ostream& out, const SimpMeshRed<N, M, T, Node, Cell, Cont>& x) {
   // Set the vertex identifiers.
   x.setNodeIdentifiers();

   writeIssVtkXml<N, M>(out, x.getVerticesBeginning(), x.getVerticesEnd(),
                        x.getIndexedSimplicesBeginning(),
                        x.getIndexedSimplicesEnd());
}


//! Write in legacy VTK unstructured grid format.
template < std::size_t N, std::size_t M, typename T,
         template<class> class Node,
         template<class> class Cell,
         template<class, class> class Cont >
inline
void
writeVtkLegacy(std::ostream& out,
               const SimpMeshRed<N, M, T, Node, Cell, Cont>& x,
               std::string title) {
   // Set the vertex identifiers.
   x.setNodeIdentifiers();

   writeIssVtkLegacy<N, M>(out, x.getVerticesBeginning(), x.getVerticesEnd(),
                           x.getIndexedSimplicesBeginning(),
                           x.getIndexedSimplicesEnd(), title);
}

} // namespace geom
}
