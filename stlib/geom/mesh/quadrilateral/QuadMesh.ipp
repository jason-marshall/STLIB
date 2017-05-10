// -*- C++ -*-

#if !defined(__geom_mesh_quadrilateral_QuadMesh_ipp__)
#error This file is an implementation detail of the class QuadMesh.
#endif

namespace stlib
{
namespace geom {


// Build from pointers to the vertices and indexed faces.
template<std::size_t N, typename T>
inline
void
QuadMesh<N, T>::
build(const SizeType numVertices, const Vertex* vertices,
      const SizeType numFaces, const IndexedFace* indexedFaces) {
   std::vector<Vertex> v(vertices, vertices + numVertices);
   _vertices.swap(v);
   std::vector<IndexedFace> i(indexedFaces, indexedFaces + numFaces);
   _indexedFaces.swap(i);
   updateTopology();
}


// Build from pointers to the vertices and indexed faces.
template<std::size_t N, typename T>
template<typename _Index>
inline
void
QuadMesh<N, T>::
build(const SizeType numVertices, const Number* coordinates,
      const SizeType numFaces, const _Index* indices) {
   _vertices.resize(numVertices);
   _indexedFaces.resize(numFaces);
   for (SizeType i = 0; i != numVertices; ++i) {
      for (SizeType n = 0; n != N; ++n) {
         _vertices[i][n] = *coordinates++;
      }
   }
   for (SizeType i = 0; i != numFaces; ++i) {
      for (SizeType m = 0; m != 4; ++m) {
         _indexedFaces[i][m] = *indices++;
      }
   }
   updateTopology();
}


// Assignment operator.
template<std::size_t N, typename T>
inline
QuadMesh<N, T>&
QuadMesh<N, T>::
operator=(const QuadMesh& other) {
   if (this != &other) {
      _vertices = other._vertices;
      _indexedFaces = other._indexedFaces;
   }
   return *this;
}

} // namespace geom
}
