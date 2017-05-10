// -*- C++ -*-

#if !defined(__geom_mesh_quadrilateral_file_io_ipp__)
#error This file is an implementation detail.
#endif

namespace stlib
{
namespace geom {

// Write an indexed simplex set in ascii format.
template<std::size_t N, typename VertForIter, typename CellForIter>
inline
void
writeQuadMeshAscii(std::ostream& out,
                   VertForIter verticesBeginning, VertForIter verticesEnd,
                   CellForIter cellsBeginning, CellForIter cellsEnd) {
   // Set the precision.
   typedef typename std::iterator_traits<VertForIter>::value_type Value;
   typedef typename std::remove_reference<Value>::type Vertex;
   typedef typename Vertex::value_type Number;
   const int oldPrecision =
      out.precision(std::numeric_limits<Number>::digits10);

   // Write the space dimension.
   out << N << "\n";

   // Write the number of vertices.
   out << std::distance(verticesBeginning, verticesEnd) << "\n";
   // Write the vertices.
   for (; verticesBeginning != verticesEnd; ++verticesBeginning) {
      out << *verticesBeginning << "\n";
   }

   // Write the number of simplices.
   out << std::distance(cellsBeginning, cellsEnd)
       << "\n";
   // Write the connectivities.
   for (; cellsBeginning != cellsEnd; ++cellsBeginning) {
      out << *cellsBeginning << "\n";
   }

   // Restore the old precision.
   out.precision(oldPrecision);
}


// Write a quad mesh in binary format.
template<std::size_t N, bool A, typename T, typename V, typename IF>
inline
void
writeBinary(std::ostream& out, const QuadMesh<N, T>& x) {
   // Write the space dimension.
   std::size_t dim = N;
   out.write(reinterpret_cast<const char*>(&dim), sizeof(std::size_t));

   // Write the vertices.
   write(out, x.getVertices());
   // Write the faces.
   write(out, x.getIndexedFaces());
}


// Write a quad mesh in binary format.
template<std::size_t N, typename VertForIter, typename CellForIter>
inline
void
writeQuadMeshBinary(std::ostream& out,
                    VertForIter verticesBeginning, VertForIter verticesEnd,
                    CellForIter cellsBeginning, CellForIter cellsEnd) {
   typedef typename std::iterator_traits<VertForIter>::value_type Vertex;
   typedef typename std::iterator_traits<CellForIter>::value_type
   IndexedFace;

   // Write the space dimension.
   std::size_t dim = N;
   out.write(reinterpret_cast<const char*>(&dim), sizeof(std::size_t));

   // Write the number of vertices.
   std::size_t sz = std::distance(verticesBeginning, verticesEnd);
   out.write(reinterpret_cast<const char*>(&sz), sizeof(std::size_t));
   // Write the vertices.
   for (; verticesBeginning != verticesEnd; ++verticesBeginning) {
      out.write(reinterpret_cast<const char*>(&*verticesBeginning),
                sizeof(Vertex));
   }

   // Write the number of cells.
   sz = std::distance(cellsBeginning, cellsEnd);
   out.write(reinterpret_cast<const char*>(&sz), sizeof(std::size_t));
   // Write the connectivities.
   for (; cellsBeginning != cellsEnd; ++cellsBeginning) {
      out.write(reinterpret_cast<const char*>(&*cellsBeginning),
                sizeof(IndexedFace));
   }
}


// Read a quad mesh in ascii format.
template<std::size_t N, typename T>
inline
void
readAscii(std::istream& in, QuadMesh<N, T>* x) {
   // Read the space dimension.
   std::size_t n;
   in >> n;
   assert(n == N);

   // Read the vertices.
   in >> x->getVertices();
   // Read the faces.
   in >> x->getIndexedFaces();
   // Update any auxilliary topological information.
   x->updateTopology();
}


// Read a quad mesh in binary format.
template<std::size_t N, typename T>
inline
void
readBinary(std::istream& in, QuadMesh<N, T>* x) {
   // Read the space dimension.
   std::size_t dim;
   in.read(reinterpret_cast<char*>(&dim), sizeof(std::size_t));
   assert(dim == N);

   // Read the vertices.
   read(in, &x->getVertices());
   // Read the faces.
   read(in, &x->getIndexedFaces());
   // Update any auxilliary topological information.
   x->updateTopology();
}

} // namespace geom
}
