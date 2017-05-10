// -*- C++ -*-

/*!
  \file IndSimpSetIncAdj.h
  \brief An indexed simplex set in N-D that optimizes simplex quality.
*/

#if !defined(__geom_IndSimpSetIncAdj_h__)
#define __geom_IndSimpSetIncAdj_h__

#include "stlib/geom/mesh/iss/IndSimpSet.h"
#include "stlib/geom/mesh/iss/simplexAdjacencies.h"
#include "stlib/geom/mesh/iss/vertexSimplexIncidence.h"
#include "stlib/geom/mesh/iss/IssiaFaceIterator.h"

namespace stlib
{
namespace geom {

//! An indexed simplex set that stores vertex-simplex incidences and simplex adjacencies.
/*!
  \param SpaceD is the space dimension.
  \param M is the simplex dimension  By default it is SpaceD.
  \param T is the number type.  By default it is double.
  \param V is the vertex type, an N-tuple of the number type.  It must be
  subscriptable.  By default it is std::array<T,SpaceD>.
  \param IS is the Indexed Simplex type, a tuple of M+1 integers.
  It must be subscriptable.  By default it is Simplex<M,std::size_t>.

  Note that the indices for indexed simplices follow the C convention of
  starting at 0.

  Consult the \ref iss page for information on using the two indexed simplex
  sets geom::IndSimpSet and geom::IndSimpSetIncAdj.

  This class derives from geom::IndSimpSet.  Any function that takes a
  geom::IndSimpSet as an argument may also take this class as an argument.
  This includes functions that build the mesh or modify the topology.
  This functionality is made possible with the update_topology() virtual
  function.  Any free function that modifies the topology of the mesh
  calls update_topology().  In the base class, the function has no effect,
  but in this class, it builds/rebuilds the vertex-simplex incidences
  and the simplex adjacencies.
*/
template < std::size_t SpaceD,
         std::size_t _M = SpaceD,
         typename T = double >
class IndSimpSetIncAdj :
   public IndSimpSet<SpaceD, _M, T> {
   //
   // The base type.
   //

public:

   //! The base type.
   typedef IndSimpSet<SpaceD, _M, T> Base;

   //
   // Enumerations.
   //

public:

   //! The space dimension.
   BOOST_STATIC_CONSTEXPR std::size_t N = Base::N;
   //! The simplex dimension.
   BOOST_STATIC_CONSTEXPR std::size_t M = Base::M;

   //
   // Public types.
   //

public:

   //
   // Inherited from IndSimpSet.
   //

   //! The number type.
   typedef typename Base::Number Number;
   //! A vertex.
   typedef typename Base::Vertex Vertex;
   //! A simplex of vertices.
   typedef typename Base::Simplex Simplex;
   //! The face of a simplex of vertices.
   typedef typename Base::SimplexFace SimplexFace;

   //! An indexed simplex.  (A simplex of indices.)
   typedef typename Base::IndexedSimplex IndexedSimplex;
   //! The face of an indexed simplex.
   typedef typename Base::IndexedSimplexFace IndexedSimplexFace;

   //! The vertex container.
   typedef typename Base::VertexContainer VertexContainer;
   //! A vertex const iterator.
   typedef typename Base::VertexConstIterator VertexConstIterator;
   //! A vertex iterator.
   typedef typename Base::VertexIterator VertexIterator;

   //! The indexed simplex container.
   typedef typename Base::IndexedSimplexContainer IndexedSimplexContainer;
   //! An indexed simplex const iterator.
   typedef typename Base::IndexedSimplexConstIterator
   IndexedSimplexConstIterator;
   //! An indexed simplex iterator.
   typedef typename Base::IndexedSimplexIterator IndexedSimplexIterator;

   //! A simplex const iterator.
   typedef typename Base::SimplexConstIterator SimplexConstIterator;

   //
   // New types.
   //

   //! The vertex-simplex incidences.
   typedef container::StaticArrayOfArrays<std::size_t> IncidenceContainer;
   //! Iterator over the vertex-simplex incidences.
   typedef typename IncidenceContainer::const_iterator IncidenceConstIterator;

   //! The simplex adjacencies.
   typedef std::vector<std::array<std::size_t, M+1> > AdjacencyContainer;

   // Faces.

   //! A face is determined by a simplex index and an integer in [0..M].
   typedef std::pair<std::size_t, std::size_t> Face;
   //! A bidirectional, iterator on the faces.
   typedef IssiaFaceIterator<IndSimpSetIncAdj> FaceIterator;

   //
   // Member data.
   //

public:

   //! The vertex-simplex incidences.
   /*! Note that for each vertex, the incident simplex indices are stored 
     in sorted order. */
   IncidenceContainer incident;
   //! The simplex adjacencies.
   AdjacencyContainer adjacent;

public:

   //--------------------------------------------------------------------------
   //! \name Constructors etc.
   //! @{

   //! Default constructor.
   IndSimpSetIncAdj() :
      Base(),
      incident(),
      adjacent() {
   }

   //! Construct from arrays of vertices and indexed simplices.
   /*!
     \param vertices is the array of vertices.
     \param indexedSimplices is the array of indexed simplices.
   */
   IndSimpSetIncAdj(const std::vector<Vertex>& vertices,
                    const std::vector<IndexedSimplex>& indexedSimplices) :
      Base(vertices, indexedSimplices),
      incident(),
      adjacent() {
      updateTopology();
   }

   //! Swap data with another mesh.
   void
   swap(IndSimpSetIncAdj& x) {
      Base::swap(x);
      incident.swap(x.incident);
      adjacent.swap(x.adjacent);
   }

   //! Construct from an indexed simplex set.
   /*!
     \param iss is the indexed simplex set.
   */
   IndSimpSetIncAdj(const Base& iss) :
      Base(iss),
      incident(),
      adjacent() {
      updateTopology();
   }

   //! Destructor.
   virtual
   ~IndSimpSetIncAdj() {
   }

   //! @}
   //--------------------------------------------------------------------------
   //! \name Simplex Adjacency Accessors
   //! @{

   //! Return the index of the n_th simplex in its m_th adjacent neighbor.
   std::size_t
   getMirrorIndex(const std::size_t n, const std::size_t m) const {
      const std::size_t a = adjacent[n][m];
      if (a == std::numeric_limits<std::size_t>::max()) {
         return std::numeric_limits<std::size_t>::max();
      }
#ifdef STLIB_DEBUG
      const std::size_t mi = std::find(adjacent[a].begin(),
                                       adjacent[a].end(), n) -
                             adjacent[a].begin();
      assert(mi < M + 1);
      return mi;
#else
      // Return the index of the value n.
      return std::find(adjacent[a].begin(),
                       adjacent[a].end(), n) -
             adjacent[a].begin();
#endif
   }

   //! @}
   //--------------------------------------------------------------------------
   //! \name Face accessors.
   //! @{

   //! Return the number of faces.
   /*!
     \note This is a slow function.  It counts the faces.
   */
   std::size_t
   computeFacesSize() const {
      return std::distance(getFacesBeginning(), getFacesEnd());
   }

   //! Return the beginning of the faces.
   FaceIterator
   getFacesBeginning() const {
      FaceIterator x(this, 0);
      return x;
   }

   //! Return the end of the faces.
   FaceIterator
   getFacesEnd() const {
      return FaceIterator(this, Base::indexedSimplices.size());
   }

   //! Return true if the face is on the boundary.
   bool
   isOnBoundary(const Face& f) const {
      return adjacent[f.first][f.second] ==
         std::numeric_limits<std::size_t>::max();
   }

   //! Return true if the face is on the boundary.
   bool
   isOnBoundary(const FaceIterator& f) const {
      return isOnBoundary(*f);
   }

   //! @}
   //--------------------------------------------------------------------------
   //! \name Other Accessors
   //! @{

   //! Return true if the vertex is on the boundary of the mesh.
   /*!
     \param index is the index of a vertex.
   */
   bool
   isVertexOnBoundary(std::size_t index) const;

   //! @}
   //--------------------------------------------------------------------------
   //! \name Simplex Manipulators
   //! @{

   //! Reverse the orientation of the n_th simplex.
   void
   reverseOrientation(const std::size_t n) {
      std::swap(Base::indexedSimplices[n][0], Base::indexedSimplices[n][1]);
      std::swap(adjacent[n][0], adjacent[n][1]);
   }

   //! @}
   //--------------------------------------------------------------------------
   //! \name Update the topology.
   //! @{

   //! Update the data structure following a change in the topology.
   /*!
     Update the vertex-simplex incidences and simplex adjacencies following
     a change in the topology.
   */
   virtual
   void
   updateTopology() {
      vertexSimplexIncidence(&incident, Base::vertices.size(),
                             Base::indexedSimplices);
      simplexAdjacencies(&adjacent, Base::indexedSimplices, incident);
   }

   //! @}
};

} // namespace geom
}

#define __geom_IndSimpSetIncAdj_ipp__
#include "stlib/geom/mesh/iss/IndSimpSetIncAdj.ipp"
#undef __geom_IndSimpSetIncAdj_ipp__

#endif
