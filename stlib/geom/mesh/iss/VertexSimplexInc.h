// -*- C++ -*-

/*!
  \file VertexSimplexInc.h
  \brief Vertex-simplex incidences in an M-D indexed simplex set.
*/

#if !defined(__geom_VertexSimplexInc_h__)
#define __geom_VertexSimplexInc_h__

#include "stlib/container/StaticArrayOfArrays.h"
#include "stlib/ext/array.h"

#include <vector>
#include <iosfwd>

namespace stlib
{
namespace geom {

//! Vertex-simplex incidences in an M-D indexed simplex set.
/*!
  \param M is the simplex dimension.

  Note that the space dimension is not relevant as this class deals only with
  topological information.
*/
template<std::size_t M>
class VertexSimplexInc {
   //
   // Private types.
   //

private:

   //! The container for the vertex-simplex incidences.
   typedef container::StaticArrayOfArrays<std::size_t> IncidenceContainer;
   //! An iterator on the vertex-simplex incidences.
   typedef typename IncidenceContainer::iterator Iterator;

   //
   // Public types.
   //

public:

   //! A const iterator on the vertex-simplex incidences.
   typedef typename IncidenceContainer::const_iterator ConstIterator;
   //! The size type.
   typedef typename IncidenceContainer::size_type SizeType;

   //
   // Data
   //

private:

   // The array of vertex-simplex incidences.
   IncidenceContainer _inc;

public:

   //--------------------------------------------------------------------------
   //! \name Constructors etc.
   // @{

   //! Default constructor.  Empty incidence data.
   VertexSimplexInc() :
      _inc() {}

   //! Construct from the number of vertices and the array of indexed simplices.
   /*!
     \c IndexedSimplex is an tuple of M+1 integers.  It must be subscriptable.
   */
   template<typename IndexedSimplex>
   VertexSimplexInc(const std::size_t numVertices,
                    const std::vector<IndexedSimplex>& simplices) :
      _inc() {
      build(numVertices, simplices);
   }

   //! Build the vertex-simplex incidences structure.
   template<typename IndexedSimplex>
   void
   build(const std::size_t numVertices,
         const std::vector<IndexedSimplex>& simplices);

   //! Copy constructor.
   VertexSimplexInc(const VertexSimplexInc& other) :
      _inc(other._inc) {}

   //! Assignment operator.
   VertexSimplexInc&
   operator=(const VertexSimplexInc& other) {
      if (this != &other) {
         _inc = other._inc;
      }
      return *this;
   }

   //! Swap data.
   void
   swap(VertexSimplexInc& x) {
      _inc.swap(x._inc);
   }

   //! Destructor.  Leave cleaning up to the containers.
   ~VertexSimplexInc() {}

   // @}
   //--------------------------------------------------------------------------
   //! \name Accessors for the whole set of incidences.
   // @{

   //! Return the number of vertices.
   SizeType
   getNumVertices() const {
      return _inc.getNumberOfArrays();
   }

   //! Return the total number of incidences.
   SizeType
   getSize() const {
      return _inc.size();
   }

   //! Return true if the total number of incidences is zero.
   bool
   isEmpty() const {
      return _inc.empty();
   }

   //! Return the size of the largest possible array.
   SizeType
   getMaxSize() const {
      return _inc.max_size();
   }

   //! Return the memory size.
   SizeType
   getMemoryUsage() const {
      return _inc.getMemoryUsage();
   }

   //! Return a const iterator to the first value.
   ConstIterator
   getBeginning() const {
      return _inc.begin();
   }

   //! Return a const iterator to one past the last value.
   ConstIterator
   getEnd() const {
      return _inc.end();
   }

   // @}
   //--------------------------------------------------------------------------
   //! \name Accessors.
   // @{

   //! Return the number of incident simplices to the n_th vertex.
   SizeType
   getSize(const std::size_t n) const {
      return _inc.size(n);
   }

   //! Return true if the n_th vertex has no incident simplices.
   bool
   isEmpty(const std::size_t n) const {
      return _inc.empty(n);
   }

   //! Return a const iterator to the first incident cell to the n_th vertex.
   ConstIterator
   getBeginning(const std::size_t n) const {
      return _inc.begin(n);
   }

   //! Return a const iterator to one past the last incident cell to the n_th vertex.
   ConstIterator
   getEnd(const std::size_t n) const {
      return _inc.end(n);
   }

   //! Return a const iterator to the first incident cell to the n_th vertex.
   ConstIterator
   operator()(const std::size_t n) const {
      return _inc(n);
   }

   //! Return a const iterator to the first incident cell to the n_th vertex.
   ConstIterator
   operator[](const std::size_t n) const {
      return _inc(n);
   }

   //! Return the m_th incident cell of the n_th vertex.
   int
   operator()(const std::size_t n, const std::size_t m) const {
      return _inc(n, m);
   }

   // @}
   //--------------------------------------------------------------------------
   //! \name Manipulators.
   // @{

   //! Clear incidences data.  Free the memory.
   void
   clear() {
      _inc.clear();
   }

   // @}
   //--------------------------------------------------------------------------
   //! \name Equality.
   //@{

   //! Return true if the incidences are the same.
   bool
   operator==(const VertexSimplexInc& x) const {
      return _inc == x._inc;
   }

   //! Return true if the incidences are not the same.
   bool
   operator!=(const VertexSimplexInc& x) const {
      return ! operator==(x);
   }

   // @}
   //--------------------------------------------------------------------------
   //! \name File I/O.
   //@{

   //! Write the vertex-simplex incidences.
   void
   put(std::ostream& out) const {
      out << _inc;
   }

   //@}
};

//
// File output.
//

//! Write the vertex-simplex incidences.
template<std::size_t M>
std::ostream&
operator<<(std::ostream& out, const VertexSimplexInc<M>& x);

} // namespace geom
}

#define __geom_VertexSimplexInc_ipp__
#include "stlib/geom/mesh/iss/VertexSimplexInc.ipp"
#undef __geom_VertexSimplexInc_ipp__

#endif
