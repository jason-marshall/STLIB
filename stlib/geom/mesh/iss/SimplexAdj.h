// -*- C++ -*-

/*!
  \file SimplexAdj.h
  \brief Simplex-simplex adjacencies in an indexed simplex set in N-D.
*/

#if !defined(__geom_SimplexAdj_h__)
#define __geom_SimplexAdj_h__

#include "stlib/geom/mesh/iss/vertexSimplexIncidence.h"

#include "stlib/geom/kernel/simplexTopology.h"

#include <vector>
#include <iosfwd>

namespace stlib
{
namespace geom {

USING_STLIB_EXT_ARRAY;

//! Simplex-simplex adjacencies in an M-D indexed simplex set.
/*!
  \param N is the simplex dimension.

  This class is used in IndSimpSetIncAdj to store the simplex-simplex
  adjacencies.  Note that the space dimension is not relevant.
  This class deals only with topological information.
*/
template<std::size_t M>
class SimplexAdj {
   //
   // Private types.
   //

private:

   //! The container for the adjacent simplex indices.
   typedef std::array < std::size_t, M + 1 > IndexContainer;

   //! The container for the simplex-simplex adjacencies.
   typedef std::vector<IndexContainer> AdjacenciesContainer;

   //
   // Data
   //

private:

   // The array of simplex-simplex adjacencies.
   AdjacenciesContainer _adj;

public:

   //--------------------------------------------------------------------------
   //! \name Constructors etc.
   // @{

   //! Default constructor.  Empty adjacency data.
   SimplexAdj() :
      _adj() {}

   //! Copy constructor.
   SimplexAdj(const SimplexAdj& other) :
      _adj(other._adj) {}

   //! Assignment operator.
   SimplexAdj&
   operator=(const SimplexAdj& other) {
      if (&other != this) {
         _adj = other._adj;
      }
      return *this;
   }

   //! Construct from the array of indexed simplices and the vertex-simplex incidences.
   SimplexAdj(const std::vector < std::array < std::size_t, M + 1 > > & simplices,
              const container::StaticArrayOfArrays<std::size_t>& vertexSimplexInc) {
      build(simplices, vertexSimplexInc);
   }

   //! Build the vertex-simplex adjacencies structure.
   void
   build(const std::vector < std::array < std::size_t, M + 1 > > & simplices,
         const container::StaticArrayOfArrays<std::size_t>& vertexSimplexInc);

   //! Construct from the number of vertices and the array of indexed simplices.
   SimplexAdj(const std::size_t numVertices,
              const std::vector < std::array < std::size_t, M + 1 > > & simplices) {
      build(numVertices, simplices);
   }

   //! Build the vertex-simplex adjacencies structure.
   void
   build(const std::size_t numVertices,
         const std::vector < std::array < std::size_t, M + 1 > > & simplices) {
      container::StaticArrayOfArrays<std::size_t> vsi;
      vertexSimplexIncidence(&vsi, numVertices, simplices);
      build(simplices, vsi);
   }

   //! Swap data.
   void
   swap(SimplexAdj& x) {
      _adj.swap(x._adj);
   }

   //! Destructor.  Leave cleaning up to the containers.
   ~SimplexAdj() {}

   // @}
   //--------------------------------------------------------------------------
   //! \name Accessors.
   // @{

   //! Return adjacencies of the n_th simplex.
   const std::array < std::size_t, M + 1 > &
   operator()(const std::size_t n) const {
      return _adj[n];
   }

   //! Return m_th adjacent simplex to the n_th simplex.
   std::size_t
   operator()(const std::size_t n, const std::size_t m) const {
      return _adj[n][m];
   }

   //! Return number of simplices.
   std::size_t
   getSize() const {
      return _adj.size();
   }

   //! Return number of simplices adjacent to the n_th simplex.
   std::size_t
   getSize(const std::size_t n) const {
      return M + 1 - std::count(_adj[n].begin(), _adj[n].end(), std::size_t(-1));
   }

   // @}
   //--------------------------------------------------------------------------
   //! \name Manipulators.
   // @{

   //! Return adjacencies of the n_th simplex.
   std::array < std::size_t, M + 1 > &
   operator()(const std::size_t n) {
      return _adj[n];
   }

   //! Set the m_th adjacent simplex to the n_th simplex.
   void
   set(const std::size_t n, const std::size_t m, const std::size_t index) {
      _adj[n][m] = index;
   }

   // @}
   //--------------------------------------------------------------------------
   //! \name Equality.
   //@{

   //! Return true if the adjacencies are the same.
   bool
   operator==(const SimplexAdj& x) const {
      return _adj == x._adj;
   }

   //! Return true if the adjacencies are not the same.
   bool
   operator!=(const SimplexAdj& x) const {
      return ! operator==(x);
   }

   // @}
   //--------------------------------------------------------------------------
   //! \name File I/O.
   //@{

   //! Write the simplex-simplex adjacencies.
   void
   put(std::ostream& out) const {
      out << _adj;
   }

   //@}
};


//
// File output.
//


//! Write the simplex adjacencies.
template<std::size_t M>
inline
std::ostream&
operator<<(std::ostream& out, const SimplexAdj<M>& x) {
   x.put(out);
   return out;
}

} // namespace geom
}

#define __geom_SimplexAdj_ipp__
#include "stlib/geom/mesh/iss/SimplexAdj.ipp"
#undef __geom_SimplexAdj_ipp__

#endif
