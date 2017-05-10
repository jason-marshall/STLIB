// -*- C++ -*-

#if !defined(__geom_ScanConversionPolygon_h__)
#define __geom_ScanConversionPolygon_h__

#include "stlib/geom/kernel/Point.h"
#include "stlib/geom/kernel/Line_2.h"
#include "stlib/geom/grid/RegularGrid.h"
#include "stlib/geom/polytope/CyclicIndex.h"

#include <iostream>
#include <vector>
#include <algorithm>

#include <cassert>
#include <cmath>
#include <cfloat>

namespace stlib
{
namespace geom
{

//! Class for a polygon in 2-D.
/*!
  \param _Index is the integer index type.
  \param T is the number type.  By default it is double.

  A ScanConversionPolygon is a list of vertices that are ordered in the positive,
  (counter-clockwise), direction.  The edges have outward normals.
*/
template < typename _Index, typename T = double >
class ScanConversionPolygon
{
public:

  //
  // Public types.
  //

  //! The integer index type.
  typedef _Index Index;
  //! The floating point number type.
  typedef T Number;

  //! The representation of a point in 2 dimensions.
  typedef std::array<Number, 2> Point;

private:

  //! Container of points.
  typedef std::vector<Point> Container;

public:

  //
  // More public types.
  //

  //! An iterator over points.
  typedef typename Container::iterator Iterator;

  //! A const Iterator over points.
  typedef typename Container::const_iterator ConstIterator;

private:

  //
  // Data
  //

  Container _vertices;

public:

  //--------------------------------------------------------------------------
  //! \name Constructors etc.
  // @{

  //! Default constructor.  Uninitialized memory.
  ScanConversionPolygon() :
    _vertices() {}

  //! Constructor.  Reserve room for size vertices.
  ScanConversionPolygon(std::size_t size);

  //! Copy constructor.
  ScanConversionPolygon(const ScanConversionPolygon& other);

  //! Assignment operator.
  ScanConversionPolygon&
  operator=(const ScanConversionPolygon& other);

  //! Trivial destructor.
  ~ScanConversionPolygon() {}

  // @}
  //--------------------------------------------------------------------------
  //! \name Accesors.
  // @{

  //! Return the number of vertices.
  std::size_t
  getVerticesSize() const
  {
    return _vertices.size();
  }

  //! Return a const reference to the specified vertex.
  const Point&
  getVertex(const std::size_t n) const
  {
    return _vertices[n];
  }

  // @}
  //--------------------------------------------------------------------------
  //! \name Manipulators.
  // @{

  // Return the beginning of the vertices.
  Iterator
  getVerticesBeginning()
  {
    return _vertices.begin();
  }

  // Return the end of the vertices.
  Iterator
  getVerticesEnd()
  {
    return _vertices.end();
  }

  //! Clear the vertices.
  void
  clear()
  {
    _vertices.clear();
  }

  //! Add a vertex.
  void
  insert(const Point& x)
  {
    _vertices.push_back(x);
  }

  // @}
  //--------------------------------------------------------------------------
  //! \name Mathematical functions.
  // @{

  //! Order the vertices in a positive orientation.
  void
  orderVertices();

  //! Remove adjacent duplicate vertices of an ordered polygon.
  void
  removeDuplicates();

  //! Find the top and bottom of the polygon. Return bottom vertex index.
  std::size_t
  computeBottomAndTop(Number* bottom, Number* top) const;

  //! Scan convert the ScanConversionPolygon in a 2-D grid.
  /*!
    \param coords is an output Iterator for the set of grid indices.
    \param extents are the extents of the grid.
  */
  template<typename IndexOutputIterator>
  void
  scanConvert(IndexOutputIterator coords,
              const std::array<std::size_t, 2>& extents) const
  {
    std::array<Index, 2> multiIndex = {{0, 0}};
    scanConvert(coords, extents, multiIndex);
  }

  //! Scan convert the ScanConversionPolygon in a 3-D grid.
  /*!
    \param coords is an output Iterator for the set of grid indices.
    \param extents are the extents of the grid.
    \param zCoordinate is the z-coordinate of the slice being scan-converted.
  */
  template<typename IndexOutputIterator>
  void
  scanConvert(IndexOutputIterator coords,
              const std::array<std::size_t, 3>& extents,
              const Index zCoordinate) const
  {
    std::array<Index, 3> multiIndex = {{0, 0, 0}};
    multiIndex[2] = zCoordinate;
    scanConvert(coords, extents, multiIndex);
  }

  //! Clip the ScanConversionPolygon against the line.
  void
  clip(const Line_2<Number>& line);

  //! Check if polygon is valid.
  /*!
    Check that the ScanConversionPolygon has at least three vertices and that adjacent
    vertices are not equal.
  */
  bool
  isValid() const;

  // @}
  //--------------------------------------------------------------------------
  //! \name File I/O.
  // @{

  //! Read the number of vertices and each vertex.
  void
  get(std::istream& in);

  //! Write each vertex.
  void
  put(std::ostream& out) const;

  //! Write a Line[] object that Mathematica can read.
  void
  mathematicaPrint(std::ostream& out) const;

  // @}

private:

  //! Scan convert the ScanConversionPolygon.
  /*!
    This function can be used for scan conversion on 2-D or 3-D grids.
    For 3-D grids, the third coordinate of \c multiIndex should be
    set to the z-coordinate of the slice being scan-converted.
  */
  template<typename IndexOutputIterator, std::size_t N>
  void
  scanConvert(IndexOutputIterator coords,
              const std::array<std::size_t, N>& extents,
              std::array<Index, N> multiIndex) const;

  //! Scan convert the triangle.
  /*!
    Add the set of coordinates of the grid points in this triangle
    to \c coords.
    Precondition:
    This polygon must have exactly three vertices.
  */
  template<typename IndexOutputIterator, std::size_t N>
  void
  scanConvertTriangle(IndexOutputIterator coords,
                      const std::array<std::size_t, N>& extents,
                      std::array<Index, N> multiIndex) const;
};


//! Return true if the polygons have the same points in the same order.
/*! \relates ScanConversionPolygon */
template<typename _Index, typename T>
inline
bool
operator==(const ScanConversionPolygon<_Index, T>& a,
           const ScanConversionPolygon<_Index, T>& b);


//! Return true if they don't have the same points in the same order.
/*! \relates ScanConversionPolygon */
template<typename _Index, typename T>
inline
bool
operator!=(const ScanConversionPolygon<_Index, T>& a,
           const ScanConversionPolygon<_Index, T>& b)
{
  return !(a == b);
}


//! Read the number of vertices and each vertex.
/*! \relates ScanConversionPolygon */
template<typename _Index, typename T>
inline
std::istream&
operator>>(std::istream& in, ScanConversionPolygon<_Index, T>& x)
{
  x.get(in);
  return in;
}


//! Write each vertex.
/*! \relates ScanConversionPolygon */
template<typename _Index, typename T>
inline
std::ostream&
operator<<(std::ostream& out, const ScanConversionPolygon<_Index, T>& x)
{
  x.put(out);
  return out;
}


} // namespace geom
}

#define __geom_ScanConversionPolygon_ipp__
#include "stlib/geom/polytope/ScanConversionPolygon.ipp"
#undef __geom_ScanConversionPolygon_ipp__

#endif
