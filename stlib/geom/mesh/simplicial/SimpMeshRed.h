// -*- C++ -*-

/*!
  \file SimpMeshRed.h
  \brief Class for a tetrahedral mesh with topological optimization capabilities.
*/

#if !defined(__geom_mesh_simplicial_SimpMeshRed_h__)
#define __geom_mesh_simplicial_SimpMeshRed_h__

#include "stlib/geom/mesh/simplicial/SmrNode.h"
#include "stlib/geom/mesh/simplicial/SmrCell.h"
#include "stlib/geom/mesh/simplicial/FaceIterator.h"
#include "stlib/geom/mesh/simplicial/EdgeIterator.h"

#include "stlib/geom/mesh/iss/IndSimpSetIncAdj.h"
#include "stlib/geom/kernel/BBox.h"

#include "stlib/ads/functor/composite_compare.h"
#include "stlib/ads/iterator/MemFunIterator.h"
#include "stlib/ads/algorithm/Triplet.h"

#include <list>
#include <set>
#include <map>

#include <cassert>

namespace stlib
{
namespace geom {

//! A simplicial mesh data structure.
/*!
  \param SpaceD is the space dimension.
  \param _M is the simplex dimension  By default it is SpaceD.
  \param T is the number type.  By default it is double.
  \param _Node is the node type.
  \param _Cell is the cell (simplex) type.
  \param Container is the container for storing the vertices and cells.
*/
template < std::size_t SpaceD,
         std::size_t _M = SpaceD,
         typename T = double,
         template<class> class _Node = SmrNode,
         template<class> class _Cell = SmrCell,
// CONTINUE
// Had to add the allocator class to appease MSVC++.
// See notes below.
         template < class _Elem,
         class = std::allocator<_Elem> > class Container =
         std::list >
class SimpMeshRed {
   //
   // Enumerations.
   //

public:

   //! The space dimension and simplex dimension.
   enum {N = SpaceD, M = _M};

   //
   // Types.
   //

private:

   // CONTINUE
   // I should be able to do this more cleanly.  See page 112 of
   // "C++ Templates, The Complete Guide".  I have to add the allocator to
   // get MSVC++ to compile it.
   typedef Container < _Node<SimpMeshRed>,
           std::allocator<_Node<SimpMeshRed> > > NodeContainer;
   typedef Container < _Cell<SimpMeshRed>,
           std::allocator<_Cell<SimpMeshRed> > > CellContainer;

public:

   //
   // Nodes.
   //

   //! A node.
   typedef typename NodeContainer::value_type Node;
   //! Node iterator.
   typedef typename NodeContainer::iterator NodeIterator;
   //! Vertex const iterator.
   typedef typename NodeContainer::const_iterator NodeConstIterator;

   //
   // Cells.
   //

   //! A cell (simplex).
   typedef typename CellContainer::value_type Cell;
   //! Cell iterator.
   typedef typename CellContainer::iterator CellIterator;
   //! Cell const iterator.
   typedef typename CellContainer::const_iterator CellConstIterator;

   //
   // Faces.
   //

   //! A const face of a cell is determined by a cell and a node index.
   typedef std::pair<CellConstIterator, std::size_t> ConstFace;
   //! A bidirectional, constant iterator on the faces.
   typedef geom::FaceIterator<M, ConstFace, CellConstIterator>
   FaceConstIterator;
   //! A face of a cell is determined by a cell and a node index.
   typedef std::pair<CellIterator, std::size_t> Face;
   //! A bidirectional, iterator on the faces.
   typedef geom::FaceIterator<M, Face, CellIterator> FaceIterator;

   //
   // Edges.
   //

   //! A const edge of a cell is determined by a cell and two node indices.
   typedef ads::Triplet<CellConstIterator, std::size_t, std::size_t> ConstEdge;
   //! A bidirectional, constant iterator on the edges.
   typedef geom::EdgeIterator<SimpMeshRed, true> EdgeConstIterator;
   //! An edge of a cell is determined by a cell and two node indices.
   typedef ads::Triplet<CellIterator, std::size_t, std::size_t> Edge;
   //! A bidirectional, iterator on the edges.
   typedef geom::EdgeIterator<SimpMeshRed, false> EdgeIterator;


   //
   // Miscellaneous.
   //

   //! The number type.
   typedef T Number;
   //! A node (a Cartesian point).
   typedef typename Node::Vertex Vertex;
   //! A bounding box.
   typedef geom::BBox<Number, N> BBox;

   //! The size type.
   typedef std::size_t SizeType;
   //! The pointer difference type.
   typedef typename NodeContainer::difference_type DifferenceType;

   //
   // Simplex
   //

   //! A simplex of indices.
   typedef std::array < std::size_t, M + 1 > IndexedSimplex;
   //! A simplex of vertices.
   typedef std::array < Vertex, M + 1 > Simplex;

   //
   // Node member function iterators.
   //

   //! Vertex point iterator.
   typedef ads::MemFunIterator<NodeConstIterator, Node, const Vertex&, true>
   VertexIterator;
   //! Node identifier iterator.
   typedef ads::MemFunIterator<NodeConstIterator, Node, std::size_t, true>
   NodeIdentifierIterator;
   //! Cell identifier iterator.
   typedef ads::MemFunIterator<CellConstIterator, Cell, std::size_t, true>
   CellIdentifierIterator;

   //
   // Indexed simplex iterator.
   //

#define __geom_mesh_simplicial_SMR_IndSimpIter_ipp__
#include "SMR_IndSimpIter.ipp"
#undef __geom_mesh_simplicial_SMR_IndSimpIter_ipp__

   //! A const iterator over indexed simplices.
   typedef IndSimpIter IndexedSimplexIterator;

   //
   // Simplex iterator.
   //

#define __geom_mesh_simplicial_SMR_SimpIter_ipp__
#include "SMR_SimpIter.ipp"
#undef __geom_mesh_simplicial_SMR_SimpIter_ipp__

   //! A const iterator over simplices.
   typedef SimpIter SimplexIterator;

   // CONTINUE: REMOVE?
   //! Functor for comparing node iterators by their identifiers.
   struct NodeIteratorCompare :
      public std::binary_function<NodeIterator, NodeIterator, bool> {
      //! Compare node iterators by their identifiers.
      bool
      operator()(const NodeIterator& x, const NodeIterator& y) const {
         return x->getIdentifier() < y->getIdentifier();
      }
   };

   //! Functor for comparing node pointers by their identifiers.
   struct NodePointerCompare :
      public std::binary_function<const Node*, const Node*, bool> {
      //! Compare node pointers by their identifiers.
      bool
      operator()(const Node* const x, const Node* const y) const {
         return x->getIdentifier() < y->getIdentifier();
      }
   };

   //! Functor for comparing cell iterators by their identifiers.
   struct CellIteratorCompare :
      public std::binary_function<CellIterator, CellIterator, bool> {
      //! Compare cell iterators by their identifiers.
      bool
      operator()(const CellIterator& x, const CellIterator& y) const {
         return x->getIdentifier() < y->getIdentifier();
      }
   };

   //! Functor for comparing faces.
   struct FaceCompare :
      public std::binary_function<Face, Face, bool> {
      //! Compare the faces.
      bool
      operator()(const Face& x, const Face& y) const {
         return x.first->getIdentifier() < y.first->getIdentifier() ||
                (x.first->getIdentifier() == y.first->getIdentifier() &&
                 x.second < y.second);
      }
   };

   //! Functor for comparing face iterators.
   struct FaceIteratorCompare :
      public std::binary_function<FaceIterator, FaceIterator, bool> {
      //! Compare the face iterators.
      bool
      operator()(const FaceIterator& x, const FaceIterator& y) const {
         return x->first->getIdentifier() < y->first->getIdentifier() ||
                (x->first->getIdentifier() == y->first->getIdentifier() &&
                 x->second < y->second);
      }
   };

   // CONTINUE: REMOVE?
   //! A set of node iterators.
   typedef std::set<NodeIterator, NodeIteratorCompare> NodeIteratorSet;
   //! A set of node pointers.
   typedef std::set<Node*, NodePointerCompare> NodePointerSet;
   //! A set of cell iterators.
   typedef std::set<CellIterator, CellIteratorCompare> CellIteratorSet;
   //! A set of faces.
   typedef std::set<Face, FaceCompare> FaceSet;
   //! A set of face iterators.
   typedef std::set<FaceIterator, FaceIteratorCompare> FaceIteratorSet;


   //
   // Data
   //

private:

   //! The nodes.
   NodeContainer _nodes;
   //! The cells.
   CellContainer _cells;

public:

   //--------------------------------------------------------------------------
   //! \name Constructors and Destructor.
   //! @{

   //! Default constructor.  Empty containers.
   SimpMeshRed() :
      _nodes(),
      _cells() {}

   //! Copy constructor.
   SimpMeshRed(const SimpMeshRed& other) :
      _nodes(other._nodes),
      _cells(other._cells) {}

   //! Construct from an indexed simplex set.
   SimpMeshRed(const IndSimpSet<N, M, Number>& iss) :
      _nodes(),
      _cells() {
      build(iss);
   }

   //! Assignment operator.
   SimpMeshRed&
   operator=(const SimpMeshRed& other) {
      if (&other != this) {
         _nodes = other._nodes;
         _cells = other._cells;
      }
      return *this;
   }

   //! Build from an indexed simplex set.
   /*!
     The value type for the vertices must be \c std::array<T,N>.
     The value type for the simplices must be subscriptable.
   */
   template<typename VertInIter, typename SimpInIter>
   void
   build(VertInIter verticesBeginning, VertInIter verticesEnd,
         SimpInIter simplicesBeginning, SimpInIter simplicesEnd);

   //! Build from an indexed simplex set.
   void
   build(const IndSimpSet<N, M, Number>& iss) {
      build(iss.vertices.begin(), iss.vertices.end(),
            iss.indexedSimplices.begin(), iss.indexedSimplices.end());
   }

   //! Swap.
   void
   swap(SimpMeshRed& x) {
      _nodes.swap(x._nodes);
      _cells.swap(x._cells);
   }

   //! Clear the mesh.
   void
   clear() {
      _nodes.clear();
      _cells.clear();
   }

   //! Destructor.
   ~SimpMeshRed() {}

   //! @}
   //--------------------------------------------------------------------------
   //! \name Dimension accessors.
   //! @{

   //! Return the space dimension.
   std::size_t
   getSpaceDimension() const {
      return N;
   }

   //! Return the simplex dimension.
   std::size_t
   getSimplexDimension() const {
      return M;
   }

   //! @}
   //--------------------------------------------------------------------------
   //! \name Node accessors.
   //! @{

   //! Return true if there are no nodes.
   bool
   areNodesEmpty() const {
      return _nodes.empty();
   }

   //! Return the number of nodes.
   /*!
     \note This is a slow function.  It counts the nodes.
   */
   SizeType
   computeNodesSize() const {
      return SizeType(_nodes.size());
   }

   //! Return the beginning of the nodes.
   NodeConstIterator
   getNodesBeginning() const {
      return _nodes.begin();
   }

   //! Return the end of the nodes.
   NodeConstIterator
   getNodesEnd() const {
      return _nodes.end();
   }

   //! Return the beginning of the node vertices.
   VertexIterator
   getVerticesBeginning() const {
      return VertexIterator(&Node::getVertex, _nodes.begin());
   }

   //! Return the end of the node vertices.
   VertexIterator
   getVerticesEnd() const {
      return VertexIterator(&Node::getVertex, _nodes.end());
   }

   //! Return the beginning of the node identifiers.
   NodeIdentifierIterator
   getNodeIdentifiersBeginning() const {
      return NodeIdentifierIterator(&Node::getIdentifier, _nodes.begin());
   }

   //! Return the end of the vertex identifiers.
   NodeIdentifierIterator
   getNodeIdentifiersEnd() const {
      return NodeIdentifierIterator(&Node::getIdentifier, _nodes.end());
   }

   // CONTINUE: I think I can remove this.
#if 0
   //! Return the maximum node identifier.
   /*!
     CONTINUE: when I implement a scheme for managing identifiers I won't need
     this.
   */
   int
   computeMaximumNodeIdentifier() const {
      if (areNodesEmpty()) {
         // Return -1 because one more than the "maximum" is then 0.
         return -1;
      }
      return *std::max_element(getNodeIdentifiersBeginning(),
                               getNodeIdentifiersEnd());
   }
#endif

   //! @}
   //--------------------------------------------------------------------------
   //! \name Cell accessors.
   //! @{

   //! Return true if there are no cells.
   bool
   areCellsEmpty() const {
      return _cells.empty();
   }

   //! Return the number of cells.
   /*!
     \note This is a slow function.  It counts the cells.
   */
   SizeType
   computeCellsSize() const {
      return SizeType(_cells.size());
   }

   //! Return the beginning of the cells.
   CellConstIterator
   getCellsBeginning() const {
      return _cells.begin();
   }

   //! Return the end of the cells.
   CellConstIterator
   getCellsEnd() const {
      return _cells.end();
   }

   //! Get the simplex given a const iterator to the cell.
   void
   getSimplex(CellConstIterator i, Simplex* s) const {
      for (std::size_t m = 0; m != M + 1; ++m) {
         (*s)[m] = i->getNode(m)->getVertex();
      }
   }

   //! Return the beginning of the cell identifiers.
   CellIdentifierIterator
   getCellIdentifiersBeginning() const {
      return CellIdentifierIterator(&Cell::getIdentifier, _cells.begin());
   }

   //! Return the end of the cell identifiers.
   CellIdentifierIterator
   getCellIdentifiersEnd() const {
      return CellIdentifierIterator(&Cell::getIdentifier, _cells.end());
   }

   // CONTINUE: I think I can remove this.
#if 0
   //! Return the maximum cell identifier.
   /*!
     CONTINUE: when I implement a scheme for managing identifiers I won't need
     this.
   */
   int
   computeMaximumCellIdentifier() const {
      if (areCellsEmpty()) {
         // Return -1 because one more than the "maximum" is then 0.
         return -1;
      }
      return *std::max_element(getCellIdentifiersBeginning(),
                               getCellIdentifiersEnd());
   }
#endif

   //! @}
   //--------------------------------------------------------------------------
   //! \name Simplex accessors.
   //! @{

   //! Return the beginning of the indexed simplices
   IndexedSimplexIterator
   getIndexedSimplicesBeginning() const {
      return IndexedSimplexIterator(getCellsBeginning());
   }

   //! Return the end of the indexed simplices
   IndexedSimplexIterator
   getIndexedSimplicesEnd() const {
      return IndexedSimplexIterator(getCellsEnd());
   }

   //! Return the beginning of the simplices
   SimplexIterator
   getSimplicesBeginning() const {
      return SimplexIterator(getCellsBeginning());
   }

   //! Return the end of the simplices
   SimplexIterator
   getSimplicesEnd() const {
      return SimplexIterator(getCellsEnd());
   }

   //! @}
   //--------------------------------------------------------------------------
   //! \name Face accessors.
   //! @{

   //! Return the number of faces.
   /*!
     \note This is a slow function.  It counts the faces.
   */
   SizeType
   computeFacesSize() const {
      return SizeType(std::distance(getFacesBeginning(), getFacesEnd()));
   }

   //! Return the beginning of the faces.
   FaceConstIterator
   getFacesBeginning() const {
      FaceConstIterator x(getCellsBeginning(), getCellsEnd());
      return x;
   }

   //! Return the end of the faces.
   FaceConstIterator
   getFacesEnd() const {
      return FaceConstIterator(getCellsEnd(), getCellsEnd());
   }

   //! @}
   //--------------------------------------------------------------------------
   //! \name Edge accessors.
   //! @{

   //! Return the number of edges.
   /*!
     \note This is a slow function.  It counts the edges.
   */
   SizeType
   computeEdgesSize() const {
      return std::distance(getEdgesBeginning(), getEdgesEnd());
   }

   //! Return the beginning of the edges.
   EdgeConstIterator
   getEdgesBeginning() const {
      EdgeConstIterator x(getCellsBeginning(), getCellsEnd());
      return x;
   }

   //! Return the end of the edges.
   EdgeConstIterator
   getEdgesEnd() const {
      return EdgeConstIterator(getCellsEnd(), getCellsEnd());
   }

   //! @}
   //--------------------------------------------------------------------------
   //! \name Node manipulators.
   //! @{

   //! Return the beginning of the nodes.
   NodeIterator
   getNodesBeginning() {
      return _nodes.begin();
   }

   //! Return the end of the nodes.
   NodeIterator
   getNodesEnd() {
      return _nodes.end();
   }

   //! Set the node identifiers.
   /*!
     \note This is a const member function because the node identifier is
     mutable.
   */
   void
   setNodeIdentifiers() const;

   //! Set the locations of the vertices.
   template<typename VertexInIter>
   void
   setVertices(VertexInIter begin, VertexInIter end) {
      for (NodeIterator i = getNodesBeginning(); i != getNodesEnd();
            ++i, ++begin) {
#ifdef STLIB_DEBUG
         assert(begin != end);
#endif
         i->setVertex(*begin);
      }
      if (begin != end) {
        throw std::runtime_error("Error in stlib::geom::SimpMeshRed::"
                                 "SetVertices(): The range of vertices has "
                                 "the wrong size.");
      }
   }

   //! @}
   //--------------------------------------------------------------------------
   //! \name Cell manipulators.
   //! @{

   //! Return the beginning of the cells.
   CellIterator
   getCellsBeginning() {
      return _cells.begin();
   }

   //! Return the end of the cells.
   CellIterator
   getCellsEnd() {
      return _cells.end();
   }

   //! Set the cell identifiers.
   /*!
     \note This is a const member function because the cell identifier is
     mutable.
   */
   void
   setCellIdentifiers() const;

   //! @}
   //--------------------------------------------------------------------------
   //! \name Face manipulators.
   //! @{

   //! Return the beginning of the faces.
   FaceIterator
   getFacesBeginning() {
      return FaceIterator(getCellsBeginning(), getCellsEnd());
   }

   //! Return the end of the faces.
   FaceIterator
   getFacesEnd() {
      return FaceIterator(getCellsEnd(), getCellsEnd());
   }

   //! @}
   //--------------------------------------------------------------------------
   //! \name Edge manipulators.
   //! @{

   //! Return the beginning of the edges.
   EdgeIterator
   getEdgesBeginning() {
      return EdgeIterator(getCellsBeginning(), getCellsEnd());
   }

   //! Return the end of the edges.
   EdgeIterator
   getEdgesEnd() {
      return EdgeIterator(getCellsEnd(), getCellsEnd());
   }

   //! @}
   //--------------------------------------------------------------------------
   //! \name Insert/erase nodes.
   //! @{

   //! Insert the node into the mesh.
   /*!
     Set the self iterator and the identifier.
   */
   NodeIterator
   insertNode(const Node& node = Node());

   //! Insert a copy of the node into the mesh.
   NodeIterator
   insertNode(const NodeIterator node) {
      return insert_vertex(*node);
   }

   //! Erase a vertex.
   /*!
     No cell should be incident to this vertex.
   */
   void
   eraseNode(const NodeIterator node) {
      _nodes.erase(node);
   }

   //! Find the node iterator for a specified pointer.

   //! Merge two nodes. Erase the second.
   /*!
     The two nodes should not have any incident cells in common.
   */
   void
   merge(Node* x, Node* y) {
#if 0
      // CONTINUE
      std::cerr << "Merge " << x->getIdentifier() << " "
                << y->getIdentifier() << "\n";
#endif
      assert(x != y);
      // Merge the vertex-simplex incidences.
      x->insertCells(y->getCellIteratorsBeginning(), y->getCellIteratorsEnd());
      // Fix the simplex-vertex incidences.
      y->replace(x);
      // Erase the second vertex.
      eraseNode(y->getSelf());
   }

   //! @}
   //--------------------------------------------------------------------------
   //! \name Insert/erase cells.
   //! @{

   //! Insert the cell into the mesh and set the self iterator and identifier.
   CellIterator
   insertCell(const Cell& c = Cell());

   //! Insert a copy of the cell into the mesh.
   CellIterator
   insertCell(const CellIterator c) {
      return insertCell(*c);
   }

   //! Erase a cell.
   /*!
     Unlink the cell and erase it from the mesh.
   */
   void
   eraseCell(const CellIterator c) {
      c->unlink();
      _cells.erase(c);
   }

   //! Erase a range of cells.
   /*!
     Unlink the cells and erase them from the mesh.

     \c InIter is an input iterator for cell iterators.
   */
   template<typename InIter>
   void
   eraseCells(InIter begin, InIter end) {
      for (; begin != end; ++begin) {
         eraseCell(*begin);
      }
   }

   //! @}

private:

   void
   buildCellAdjacencies();

};

} // namespace geom
}

#define __geom_mesh_simplicial_SimpMeshRed_ipp__
#include "stlib/geom/mesh/simplicial/SimpMeshRed.ipp"
#undef __geom_mesh_simplicial_SimpMeshRed_ipp__

#endif
