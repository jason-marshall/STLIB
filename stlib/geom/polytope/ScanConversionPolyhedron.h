// -*- C++ -*-

/*!
  \file ScanConversionPolyhedron.h
  \brief A class for a polyhedron in 3-D designed for scan conversion.
*/

#if !defined(__geom_ScanConversionPolyhedron_h__)
#define __geom_ScanConversionPolyhedron_h__

#include "stlib/geom/polytope/IndexedEdgePolyhedron.h"
#include "stlib/geom/polytope/ScanConversionPolygon.h"

#include "stlib/geom/kernel/SegmentMath.h"
#include "stlib/geom/grid/RegularGrid.h"

#include <iostream>
#include <iomanip>
#include <vector>

#include <cassert>
#include <cmath>

namespace stlib
{
namespace geom
{

//! A class for a polyhedron in 3-D designed for scan conversion.
/*!
  \param _Index is the integer index type.
  \param T is the number type.  By default it is double.

  As the name suggests, this class is designed for 3-D polyhedron
  scan conversion.  The Polyhedron is represented as a set of edges.
  The edges support mathematical operations that enable efficient slicing
  to obtain polygons.  This, in turn, enables efficient scan conversion.
*/
template < typename _Index, typename T = double >
class ScanConversionPolyhedron
{
  //
  // Public types.
  //

public:

  //! The integer index type.
  typedef _Index Index;
  //! The floating point number type.
  typedef T Number;
  //! The representation of a point in 3 dimensions.
  typedef std::array<Number, 3> Point;
  //! A line segment that supports mathematical operations.
  typedef SegmentMath<3, Number> Segment;

  //
  // Private types.
  //

private:

  typedef ScanConversionPolygon<Index, Number> Polygon;

  //
  // Data
  //

private:

  //! The edges of the polyhedron.
  std::vector<Segment> _edges;

public:

  //--------------------------------------------------------------------------
  //! \name Constructors etc.
  // @{

  //! Default constructor.  Uninitialized memory.
  ScanConversionPolyhedron() :
    _edges() {}

  //! Copy constructor.
  ScanConversionPolyhedron(const ScanConversionPolyhedron& other) :
    _edges(other._edges) {}

  //! Assignment operator.
  ScanConversionPolyhedron&
  operator=(const ScanConversionPolyhedron& other);

  //! Assignment operator from an IndexedEdgePolyhedron.
  ScanConversionPolyhedron&
  operator=(const IndexedEdgePolyhedron<Number>& x);

  //! Trivial destructor.
  ~ScanConversionPolyhedron() {}

  // @}
  //--------------------------------------------------------------------------
  //! \name Mathematical functions.
  // @{

  //! Make a BBox containing the polyhedron.
  void
  computeBBox(BBox<Number, 3>* bb) const;

  //! Convert the Cartesian coordinates of this polyhedron to index coordinates.
  void
  convertLocationsToIndices(const RegularGrid<3, Number>& grid);

  //! Scan convert the polyhedron.
  /*!
    \param coordinates is an output iterator for the set of coordinates inside.
    \param grid describes the grid on which to perform the scan conversion.
  */
  template<typename IndexOutputIterator>
  void
  scanConvert(IndexOutputIterator coordinates,
              const RegularGrid<3, Number>& grid) const;

  // @}
  //--------------------------------------------------------------------------
  //! \name Accesors.
  // @{

  //! Return a const reference to the edges.
  const std::vector<Segment>&
  getEdges() const
  {
    return _edges;
  }

  // @}
  //--------------------------------------------------------------------------
  //! \name Manipulators.
  // @{

  //! Clear the edges.
  void
  clear()
  {
    _edges.clear();
  }

  //! Add an edge to the polyhedron.
  void
  insertEdge(const Point& p, const Point& q);

  // @}

private:

  // Intersect with the plane of constant z value to obtain a polygon.
  void
  computeZIntersection(Polygon* polygon, Number z) const;

};


//-----------------------------------------------------------------------------
// File I/O Operators
//-----------------------------------------------------------------------------


//! Write the edges.
/*! \relates ScanConversionPolyhedron */
template<typename _Index, typename T>
std::ostream&
operator<<(std::ostream& out,
           const ScanConversionPolyhedron<_Index, T>& polyhedron);


//! Write the edges in Mathematica readable format.
/*! \relates ScanConversionPolyhedron */
template<typename _Index, typename T>
void
mathematicaPrint(std::ostream& out,
                 const ScanConversionPolyhedron<_Index, T>& polyhedron);


//! Read as a list of edges.
/*! \relates ScanConversionPolyhedron */
template<typename _Index, typename T>
std::istream&
operator>>(std::istream& in, ScanConversionPolyhedron<_Index, T>& polyhedron);



//-----------------------------------------------------------------------------
// Equality Operators
//-----------------------------------------------------------------------------

//! Return true if the polyhedra are equal.
/*! \relates ScanConversionPolyhedron */
template<typename _Index, typename T>
inline
bool
operator==(const ScanConversionPolyhedron<_Index, T>& a,
           const ScanConversionPolyhedron<_Index, T>& b)
{
  return a.getEdges() == b.getEdges();
}


//! Return true if the polyhedra are not equal.
/*! \relates ScanConversionPolyhedron */
template<typename _Index, typename T>
inline
bool
operator!=(const ScanConversionPolyhedron<_Index, T>& a,
           const ScanConversionPolyhedron<_Index, T>& b)
{
  return !(a == b);
}


} // namespace geom
}

#define __geom_ScanConversionPolyhedron_ipp__
#include "stlib/geom/polytope/ScanConversionPolyhedron.ipp"
#undef __geom_ScanConversionPolyhedron_ipp__

#endif
