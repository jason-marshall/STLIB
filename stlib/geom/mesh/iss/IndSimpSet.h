// -*- C++ -*-

/*!
  \file IndSimpSet.h
  \brief Implements a mesh that stores vertices and indexed simplices.
*/

#if !defined(__geom_IndSimpSet_h__)
#define __geom_IndSimpSet_h__

#include "stlib/geom/mesh/iss/SimplexIterator.h"

#include <boost/config.hpp>

#include <array>
#include <unordered_map>
#include <vector>

#include <cassert>

namespace stlib
{
namespace geom {

//! Class for a mesh that stores vertices and indexed simplices.
/*!
  \param SpaceD is the space dimension.
  \param M is the simplex dimension. By default it is N.
  \param T is the number type. By default it is double.

  Note that the indices for indexed simplices follow the C convention of
  starting at 0.

  Consult the \ref iss page for information on using the two indexed simplex
  sets geom::IndSimpSet and geom::IndSimpSetIncAdj.
*/
template < std::size_t SpaceD, std::size_t _M = SpaceD, typename T = double >
class IndSimpSet {
   //
   // Enumerations.
   //

public:

   //! The space dimension.
   BOOST_STATIC_CONSTEXPR std::size_t N = SpaceD;
   //! The simplex dimension.
   BOOST_STATIC_CONSTEXPR std::size_t M = _M;

   //
   // Public types.
   //

public:

   //! The number type.
   typedef T Number;
   //! A vertex.
   typedef std::array<T, N> Vertex;
   //! A simplex of vertices.
   typedef std::array<Vertex, M + 1> Simplex;
   //! The face of a simplex of vertices.
   typedef std::array<Vertex, M> SimplexFace;

   //! An indexed simplex. (A simplex of indices.)
   typedef std::array<std::size_t, M + 1> IndexedSimplex;
   //! The face of an indexed simplex.
   typedef std::array<std::size_t, M> IndexedSimplexFace;

   //! The vertex container.
   typedef std::vector<Vertex> VertexContainer;
   //! A vertex const iterator.
   typedef typename VertexContainer::const_iterator VertexConstIterator;
   //! A vertex iterator.
   typedef typename VertexContainer::iterator VertexIterator;

   //! The indexed simplex container.
   typedef std::vector<IndexedSimplex> IndexedSimplexContainer;
   //! An indexed simplex const iterator.
   typedef typename IndexedSimplexContainer::const_iterator
   IndexedSimplexConstIterator;
   //! An indexed simplex iterator.
   typedef typename IndexedSimplexContainer::iterator
   IndexedSimplexIterator;

   //! A simplex const iterator.
   typedef SimplexIterator<IndSimpSet> SimplexConstIterator;

public:

   //
   // Data
   //

   //! The vertices.
   VertexContainer vertices;

   //! An indexed simplex is determined by the indices of M+1 vertices.
   IndexedSimplexContainer indexedSimplices;

   //--------------------------------------------------------------------------
   /*! \name Constructors etc.
     Suppose that we are dealing with a tetrahedron mesh in 3-D. Below
     we instantiate a mesh that allocates its own memory for the vertices
     and indexed simplices.
     \code
     geom::IndSimpSet<3,3> mesh;
     \endcode
     We can construct the mesh from vertices and indexed simplices stored in
     vectors:
     \code
     typedef geom::IndSimpSet<3,3> ISS;
     typedef typename ISS:Vertex Vertex;
     typedef typename ISS:IndexedSimplex IndexedSimplex;
     std::vector<Vertex> vertices(numberOfVertices);
     std::vector<IndexedSimplex> indexedSimplices(numberOfSimplices);
     ...
     geom::IndSimpSet<3,3> mesh(vertices, indexedSimplices);
     \endcode
     or use C arrays:
     \code
     double* vertices = new double[3 * numberOfVertices]
     std::size_t* indexedSimplices = new std::size_t[4 * numberOfSimplices];
     ...
     geom::IndSimpSet<3,3> mesh;
     mesh.build(numberOfVertices, vertices, numberOfSimplices, simplices);
     \endcode
   */
   //! @{

public:

   //! Default constructor. Empty simplex set.
   IndSimpSet() :
      vertices(),
      indexedSimplices() {
   }

   //! Construct from arrays of vertices and indexed simplices.
   /*!
     \param vertices is the array of vertices.
     \param indexedSimplices is the array of indexed simplices.
   */
   IndSimpSet(const std::vector<Vertex>& vertices_,
              const std::vector<IndexedSimplex>& indexedSimplices_) :
      vertices(vertices_),
      indexedSimplices(indexedSimplices_) {
   }

   //! Swap data with another mesh.
   void
   swap(IndSimpSet& x) {
      vertices.swap(x.vertices);
      indexedSimplices.swap(x.indexedSimplices);
   }

   //! Destructor.
   virtual
   ~IndSimpSet() {}

   //! Convert identifier simplices to index simplices.
   void
   convertFromIdentifiersToIndices
   (const std::vector<std::size_t>& vertexIdentifiers);

   //! @}
   //--------------------------------------------------------------------------
   //! \name Simplex Accessors
   //! @{

public:

   //! Return true if there are no vertices or simplices.
   std::size_t
   empty() const {
      return vertices.empty() && indexedSimplices.empty();
   }

   //! Return a const iterator to the beginning of the simplices.
   SimplexConstIterator
   getSimplicesBegin() const {
      return SimplexConstIterator(*this);
   }

   //! Return a const iterator to the end of the simplices.
   SimplexConstIterator
   getSimplicesEnd() const {
      SimplexConstIterator i(*this);
      i += indexedSimplices.size();
      return i;
   }

   //! Return a const reference to the m_th vertex of the n_th simplex.
   const Vertex&
   getSimplexVertex(const std::size_t n, const std::size_t m) const {
      return vertices[indexedSimplices[n][m]];
   }

   //! Get the n_th simplex.
   void
   getSimplex(const std::size_t n, Simplex* s) const {
      getSimplex(indexedSimplices.begin() + n, s);
   }

   //! Get the simplex given an iterator to the indexed simplex.
   void
   getSimplex(IndexedSimplexConstIterator i, Simplex* s) const {
      for (std::size_t m = 0; m != M + 1; ++m) {
         (*s)[m] = vertices[(*i)[m]];
      }
   }

   //! @}
   //--------------------------------------------------------------------------
   //! \name Manipulators.
   //! @{

public:

   //! Clear the mesh.
   void
   clear() {
      vertices.clear();
      indexedSimplices.clear();
      updateTopology();
   }

   //! Update the data structure following a change in the topology.
   /*!
     For this class, this function does nothing. For derived classes,
     it updates data structures that hold auxillary topological information.
   */
   virtual
   void
   updateTopology() {}

   //! @}
};

} // namespace geom
}

#define __geom_IndSimpSet_ipp__
#include "stlib/geom/mesh/iss/IndSimpSet.ipp"
#undef __geom_IndSimpSet_ipp__

#endif
