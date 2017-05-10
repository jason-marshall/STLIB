// -*- C++ -*-

/*!
  \file QuadMesh.h
  \brief Implements a mesh that stores vertices and indexed faces.
*/

#if !defined(__geom_QuadMesh_h__)
#define __geom_QuadMesh_h__

#include <array>

#include <vector>

#include <cassert>

namespace stlib
{
namespace geom {

//! Class for a mesh that stores vertices and indexed quadrilaterals.
/*!
  \param SpaceD is the space dimension.
  \param T is the number type.  By default it is double.

  Note that the indices for indexed faces follow the C convention of
  starting at 0.

  The free functions that operate on this class are grouped into the
  following categories:
  - \ref geom_mesh_quadrilateral_convert
*/
template < std::size_t SpaceD, typename T = double >
class QuadMesh {
   //
   // Enumerations.
   //

public:

   //! The space dimension and allocation.
   enum {N = SpaceD};

   //
   // Public types.
   //

public:

   //! The number type.
   typedef T Number;
   //! A vertex.
   typedef std::array<T, SpaceD> Vertex;
   //! A face of vertices.
   typedef std::array<Vertex, 4> Face;

   //! An indexed face.  (A face of indices.)
   typedef std::array<std::size_t, 4> IndexedFace;

   //! The vertex container.
   typedef std::vector<Vertex> VertexContainer;
   //! A vertex const iterator.
   typedef typename VertexContainer::const_iterator VertexConstIterator;
   //! A vertex iterator.
   typedef typename VertexContainer::iterator VertexIterator;

   //! The indexed face container.
   typedef std::vector<IndexedFace> IndexedFaceContainer;
   //! An indexed face const iterator.
   typedef typename IndexedFaceContainer::const_iterator
   IndexedFaceConstIterator;
   //! An indexed face iterator.
   typedef typename IndexedFaceContainer::iterator IndexedFaceIterator;

   //! The size type.
   typedef typename VertexContainer::size_type SizeType;

private:

   //
   // Data
   //

   //! The vertices.
   VertexContainer _vertices;

   //! An indexed face is determined by the indices of 4 vertices.
   IndexedFaceContainer _indexedFaces;

public:

   //--------------------------------------------------------------------------
   /*! \name Constructors etc.
     Suppose that we are dealing with a quadrilateral mesh in 3-D.  Below
     we instantiate a mesh that allocates its own memory for the vertices
     and indexed faces.
     \code
     geom::QuadMesh<3> mesh;
     \endcode
     We can construct the mesh from vertices and indexed faces stored in
     ADS arrays:
     \code
     typedef geom::QuadMesh<3> Mesh;
     typedef typename Mesh:Vertex Vertex;
     typedef typename Mesh:IndexedFace IndexedFace;
     std::vector<Vertex> vertices(numberOfVertices);
     std::vector<IndexedFace> indexedFaces(numberOfFaces);
     ...
     geom::QuadMesh<3> mesh(vertices, indexedFaces);
     \endcode
     or use C arrays:
     \code
     double* vertices = new[3 * numberOfVertices]
     int* indexedFaces = new[4 * numberOfFaces];
     ...
     geom::QuadMesh<3> mesh(numberOfVertices, vertices, numberOfFaces, indexedFaces);
     \endcode
   */
   //! @{

   //! Default constructor.  Empty mesh.
   QuadMesh() :
      _vertices(),
      _indexedFaces() {}

   //! Construct from arrays of vertices and indexed faces.
   /*!
     \param vertices is the array of vertices.
     \param indexedFaces is the array of indexed faces.
   */
   QuadMesh(const std::vector<Vertex>& vertices,
            const std::vector<IndexedFace>& indexedFaces) :
      _vertices(vertices),
      _indexedFaces(indexedFaces) {}

   //! Build from arrays of vertices and indexed faces.
   /*!
     Performs same actions as the constructor.

     \param vertices is the array of vertices.
     \param indexedFaces is the array of indexed faces.
   */
   void
   build(const std::vector<Vertex>& vertices,
         const std::vector<IndexedFace>& indexedFaces) {
      _vertices = vertices;
      _indexedFaces = indexedFaces;
      updateTopology();
   }

   //! Construct from pointers to the vertices and indexed faces.
   /*!
     \param numVertices is the number of vertices.
     \param vertices points to the data for the vertices.
     \param numFaces is the number of faces.
     \param indexedFaces points to the data for the indexed faces.
   */
   QuadMesh(const SizeType numVertices, const Vertex* vertices,
            const SizeType numFaces, const IndexedFace* indexedFaces) :
      _vertices(vertices, vertices + numVertices),
      _indexedFaces(indexedFaces, indexedFaces + numFaces) {}

   //! Build from pointers to the vertices and indexed faces.
   /*!
     Performs same actions as the constructor.

     \param numVertices is the number of vertices.
     \param vertices points to the data for the vertices.
     \param numFaces is the number of faces.
     \param indexedFaces points to the data for the indexed faces.
   */
   void
   build(const SizeType numVertices, const Vertex* vertices,
         const SizeType numFaces, const IndexedFace* indexedFaces);

   //! Construct from pointers to the vertices and indexed faces.
   /*!
     \param numVertices is the number of vertices.
     \param coordinates points to the data for the vertex coordinates.
     \param numFaces is the number of faces.
     \param indices points to the data for the face indices.
   */
   template<typename _Index>
   QuadMesh(const SizeType numVertices, const Number* coordinates,
            const SizeType numFaces, const _Index* indices) :
      _vertices(numVertices),
      _indexedFaces(numFaces) {
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
   }

   //! Build from pointers to the vertices and indexed faces.
   /*!
     Performs same actions as the constructor.

     \param numVertices is the number of vertices.
     \param coordinates points to the data for the vertex coordinates.
     \param numFaces is the number of faces.
     \param indices points to the data for the face indices.
   */
   template<typename _Index>
   void
   build(const SizeType numVertices, const Number* coordinates,
         const SizeType numFaces, const _Index* indices);

   //! Construct from the number of vertices and faces.
   /*!
     The vertices and indexed faces are left uninitialized.

     \param numVertices is the number of vertices.
     \param numFaces is the number of faces.
   */
   QuadMesh(const SizeType numVertices, const SizeType numFaces) :
      _vertices(numVertices),
      _indexedFaces(numFaces) {}

   //! Build from the number of vertices and faces.
   /*!
     Performs same actions as the constructor.

     \param numVertices is the number of vertices.
     \param numFaces is the number of faces.
   */
   void
   build(const SizeType numVertices, const SizeType numFaces) {
      _vertices.resize(numVertices);
      _indexedFaces.resize(numFaces);
   }

   //! Swap data with another mesh.
   void
   swap(QuadMesh& x) {
      _vertices.swap(x._vertices);
      _indexedFaces.swap(x._indexedFaces);
   }

   //! Copy constructor.
   QuadMesh(const QuadMesh& other) :
      _vertices(other._vertices),
      _indexedFaces(other._indexedFaces) {}

   //! Assignment operator.
   QuadMesh&
   operator=(const QuadMesh& other);

   //! Destructor.
   virtual
   ~QuadMesh() {}

   //! @}
   //--------------------------------------------------------------------------
   //! \name Vertex Accessors
   //! @{

   //! Return the dimension of the space.
   int
   getSpaceDimension() const {
      return N;
   }

   //! Return the number of vertices.
   SizeType
   getVerticesSize() const {
      return _vertices.size();
   }

   //! Return true if there are no vertices.
   SizeType
   areVerticesEmpty() const {
      return getVerticesSize() == 0;
   }

   //! Return a const iterator to the beginning of the vertices.
   VertexConstIterator
   getVerticesBeginning() const {
      return _vertices.begin();
   }

   //! Return a const iterator to the end of the vertices.
   VertexConstIterator
   getVerticesEnd() const {
      return _vertices.end();
   }

   //! Return a const reference to the n_th vertex.
   const Vertex&
   getVertex(const int n) const {
      return _vertices[n];
   }

   //! Return a const reference to the vertex container.
   const VertexContainer&
   getVertices() const {
      return _vertices;
   }

   //! @}
   //--------------------------------------------------------------------------
   //! \name Face Accessors
   //! @{

   //! Return the number of faces.
   SizeType
   getFacesSize() const {
      return _indexedFaces.size();
   }

   //! Return true if there are no faces.
   SizeType
   areFacesEmpty() const {
      return getFacesSize() == 0;
   }

   //! Return true if there are no vertices or faces.
   SizeType
   isEmpty() const {
      return areVerticesEmpty() && areFacesEmpty();
   }

   //! Return a const iterator to the beginning of the indexed faces.
   IndexedFaceConstIterator
   getIndexedFacesBeginning() const {
      return _indexedFaces.begin();
   }

   //! Return a const iterator to the end of the indexed faces.
   IndexedFaceConstIterator
   getIndexedFacesEnd() const {
      return _indexedFaces.end();
   }

   //! Return a const reference to the n_th indexed face.
   const IndexedFace&
   getIndexedFace(const int n) const {
      return _indexedFaces[n];
   }

   //! Return a const reference to the indexed face container.
   const IndexedFaceContainer&
   getIndexedFaces() const {
      return _indexedFaces;
   }

   //! Return a const reference to the m_th vertex of the n_th face.
   const Vertex&
   getFaceVertex(const int n, const int m) const {
      return _vertices[_indexedFaces[n][m]];
   }

   //! @}
   //--------------------------------------------------------------------------
   //! \name Vertex Manipulators
   //! @{

   //! Return an iterator to the beginning of the vertices.
   VertexIterator
   getVerticesBeginning() {
      return _vertices.begin();
   }

   //! Return an iterator to the end of the vertices.
   VertexIterator
   getVerticesEnd() {
      return _vertices.end();
   }

   //! Set the specified vertex.
   void
   setVertex(const int n, const Vertex& vertex) {
      _vertices[n] = vertex;
   }

   //! Return a reference to the vertex container.
   VertexContainer&
   getVertices() {
      return _vertices;
   }

   //! @}
   //--------------------------------------------------------------------------
   //! \name Face Manipulators
   //! @{

   //! Return an iterator to the beginning of the indexed faces.
   IndexedFaceIterator
   getIndexedFacesBeginning() {
      return _indexedFaces.begin();
   }

   //! Return an iterator to the end of the indexed faces.
   IndexedFaceIterator
   getIndexedFacesEnd() {
      return _indexedFaces.end();
   }

   //! Return a reference to the indexed face container.
   IndexedFaceContainer&
   getIndexedFaces() {
      return _indexedFaces;
   }

   //! @}
   //--------------------------------------------------------------------------
   //! \name Update the topology.
   //! @{

   //! Update the data structure following a change in the topology.
   /*!
     For this class, this function does nothing.  For derived classes,
     it updates data structures that hold auxillary topological information.
   */
   virtual
   void
   updateTopology() {}

   //! @}
};

} // namespace geom
}

#define __geom_mesh_quadrilateral_QuadMesh_ipp__
#include "stlib/geom/mesh/quadrilateral/QuadMesh.ipp"
#undef __geom_mesh_quadrilateral_QuadMesh_ipp__

#endif
