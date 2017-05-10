// -*- C++ -*-

/*!
  \file geom/mesh/simplicial/EdgeRemoval.h
  \brief Class for edge removal in a tetrahedral mesh.
*/

#if !defined(__geom_mesh_simplicial_EdgeRemoval_h__)
#define __geom_mesh_simplicial_EdgeRemoval_h__

#include "stlib/geom/mesh/simplex/SimplexModCondNum.h"

#include "stlib/container/MultiArray.h"

#include <cassert>

namespace stlib
{
namespace geom {

//! Edge removal in a tetrahedral mesh.
template < class _QualityMetric = SimplexModCondNum<3>,
         class _Point = std::array<double, 3>,
         typename NumberT = typename _Point::value_type >
class EdgeRemoval {

public:

   //
   // Public types.
   //

   //! The tetrahedron quality metric.
   typedef _QualityMetric QualityMetric;
   //! The point type.
   typedef _Point Point;
   //! The number type;
   typedef NumberT Number;

private:

   //
   // Private types.
   //

   typedef typename QualityMetric::Simplex Simplex;
   typedef typename container::MultiArray<Number, 2>::Index Index;

   //
   // Data
   //

   // The source and target of the edge.
   Point _source, _target;
   // The ring of vertices around the edge. Let there be N vertices in the ring.
   std::vector<Point> _ring;
   // An N-1 x N table for calculating the quality of the tetrahedralization.
   container::MultiArray<Number, 2> _quality;
   // An N-1 x N table for calculating the indices of the tetrahedralization.
   container::MultiArray<std::size_t, 2> _index;
   // The triangle indices of the triangulation of the ring.
   std::vector<std::array<std::size_t, 3> > _triangles;
   // The quality function.
   mutable QualityMetric _qualityFunction;

   //
   // Not implemented.
   //

   // Copy constructor not implemented.
   EdgeRemoval(const EdgeRemoval&);

   // Assignment operator.
   EdgeRemoval&
   operator=(const EdgeRemoval&);

public:

   //--------------------------------------------------------------------------
   //! \name Constructors and Destructor.
   //! @{

   //! Default constructor.  Unititialized memory.
   EdgeRemoval() :
      _source(),
      _target(),
      _ring(),
      _quality(),
      _index(),
      _triangles(),
      _qualityFunction() {}

   //! @}
   //--------------------------------------------------------------------------
   //! \name Accessors.
   //! @{

   //! Return the number of triangles in the triangulation of the ring.
   std::size_t
   getNumberOfTriangles() const {
      return _triangles.size();
   }

   //! Return the n_th triangle of the triangulation of the ring.
   const std::array<std::size_t, 3>&
   getTriangle(const std::size_t n) const {
      return _triangles[n];
   }

   //! @}
   //--------------------------------------------------------------------------
   //! \name Manipulators.
   //! @{

   //! Set the source vertex of the edge.
   void
   setSource(const Point& src) {
      _source = src;
   }

   //! Set the target vertex of the edge.
   void
   setTarget(const Point& tgt) {
      _target = tgt;
   }

   //! Set the ring of vertices around the edge.
   /*!
     The vertices should go around the edge in the positive direction.
   */
   template<typename InputIterator>
   void
   setRing(InputIterator begin, InputIterator end) {
      _ring.clear();
      _ring.insert(_ring.end(), begin, end);
      assert(_ring.size() >= 3);
      // Make sure that the quality and triangulation arrays are big enough.
      if (_quality.extents()[1] < _ring.size()) {
        _quality.rebuild(std::array<std::size_t, 2>
                         {{_ring.size() - 2, _ring.size()}});
         _index.rebuild(std::array<std::size_t, 2>
                        {{_ring.size() - 2, _ring.size()}});
      }
   }

   //! @}
   //--------------------------------------------------------------------------
   //! \name Tetrahedralization.
   //! @{

   //! Try to find a tetrahedralization with better quality that \c threshhold.
   /*!
     \return true if there is a tetrahedralization with better quality than
     the given threshhold.  Otherwise return false.
   */
   bool
   solve();

   //! @}

   //
   // Private member functions.
   //
   // CONTINUE: Should these be private?

   //! CONTINUE
   void
   fillTables();

   //! CONTINUE
   void
   buildTriangles();

   //! CONTINUE
   void
   buildTrianglesRecurse(const std::size_t i, const std::size_t j);

   // Return the worse quality of the tetrahedra:
   // _ring[i], _ring[k], _ring[j], _target
   // and
   // _source, _ring[i], _ring[k], _ring[j]
   //! CONTINUE
   Number
   computeQuality(const std::size_t i, const std::size_t k, const std::size_t j)
   const;

   // Return the quality of the complex with the center edge.
   // The quality of the complex is the quality of the worst tetrahedron.
   //! CONTINUE
   Number
   computeQualityWithEdge() const;

   // Return the quality of the tetrahedra:
   // _source, _target, _ring[i], _ring[i+1]
   //! CONTINUE
   Number
   computeQuality(const std::size_t i) const;
};


} // namespace geom
}

#define __geom_mesh_simplicial_EdgeRemoval_ipp__
#include "stlib/geom/mesh/simplicial/EdgeRemoval.ipp"
#undef __geom_mesh_simplicial_EdgeRemoval_ipp__

#endif
