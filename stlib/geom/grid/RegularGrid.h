// -*- C++ -*-

/*!
  \file RegularGrid.h
  \brief A class for a regular grid.
*/

#if !defined(__geom_RegularGrid_h__)
#define __geom_RegularGrid_h__

#include <iosfwd>

#include "stlib/geom/kernel/BBox.h"

#include "stlib/container/MultiIndexTypes.h"
#include "stlib/container/MultiIndexRange.h"

#include <limits>

namespace stlib
{
namespace geom
{

//! A regular grid in N-D.
/*!
  \param N is the dimension.
  \param T is the number type.  By default it is double.
*/
template < std::size_t N, typename T = double >
class RegularGrid
{
  //
  // Public types.
  //

private:

  //! A list of indices.
  typedef typename container::MultiIndexTypes<N>::IndexList IndexList;
  //! The index type.
  typedef typename container::MultiIndexTypes<N>::Index Index;

public:

  //! The number type.
  typedef T Number;
  //! The size type.
  typedef std::size_t SizeType;
  //! The extents of a multi-array.
  typedef typename container::MultiIndexTypes<N>::SizeList SizeList;
  //! A multi-index range.
  typedef container::MultiIndexRange<N> Range;
  //! The point type.
  typedef std::array<Number, N> Point;
  //! A bounding box.
  typedef geom::BBox<T, N> BBox;

  //
  // Data
  //

private:

  //! Extents of the grid.
  SizeList _extents;
  //! The domain spanned by the grid.
  BBox _domain;
  //! Lengths of the sides of the box.
  Point _length;
  //! Grid spacing.
  Point _delta;
  //! The inverse of the grid spacing.
  /*! Store this redundant data to avoid costly divisions. */
  Point _inverseDelta;
  //! Epsilon in index coordinates.
  Number _indexEpsilon;
  //! Epsilon in cartesian coordinates.
  Number _cartesianEpsilon;


public:

  //--------------------------------------------------------------------------
  /*! \name Constructors etc.
    We use the default copy constructor, assignment operator, and destructor.
  */
  // @{

  //! Default constructor.  Uninitialized memory.
  RegularGrid() :
    _extents(),
    _domain(),
    _length(),
    _delta(),
    _inverseDelta(),
    _indexEpsilon(),
    _cartesianEpsilon() {}

  //! Construct from grid dimensions and a Cartesian domain.
  /*!
    Construct a regular grid given the grid extents and the Cartesian domain
    that the grid spans.

    \param extents the number of grid points in each direction.
    \param domain the Cartesian domain spanned by the grid.
  */
  RegularGrid(const SizeList& extents, const BBox& domain);

  // @}
  //--------------------------------------------------------------------------
  //! \name Accesors.
  // @{

  //! Return the grid dimensions.
  const SizeList&
  getExtents() const
  {
    return _extents;
  }

  //! Return the grid spacings.
  const Point&
  getDelta() const
  {
    return _delta;
  }

  //! Return the domain spanned by the grid.
  const BBox&
  getDomain() const
  {
    return _domain;
  }

  //! Return the index epsilon.
  Number
  getIndexEpsilon() const
  {
    return _indexEpsilon;
  }

  //! Return the Cartesian epsilon.
  Number
  getCartesianEpsilon() const
  {
    return _cartesianEpsilon;
  }

  // @}
  //--------------------------------------------------------------------------
  //! \name Mathematical functions.
  // @{

  //! Convert a Cartesian coordinate to a grid index coordinate.
  void
  convertLocationToIndex(Point* p) const
  {
    for (std::size_t i = 0; i != p->size(); ++i) {
      (*p)[i] = ((*p)[i] - _domain.lower[i]) * _inverseDelta[i];
    }
  }

  //! Convert a grid index coordinate to a Cartesian coordinate.
  void
  convertIndexToLocation(Point* p) const
  {
    for (std::size_t i = 0; i != p->size(); ++i) {
      (*p)[i] = _domain.lower[i] + (*p)[i] * _delta[i];
    }
  }

  //! Convert a grid index coordinate to a Cartesian coordinate.
  void
  convert(const IndexList& index, Point* p) const
  {
    for (std::size_t i = 0; i != p->size(); ++i) {
      (*p)[i] = _domain.lower[i] + index[i] * _delta[i];
    }
  }

  //! Convert a vector in Cartesian coordinates to index coordinates.
  void
  convertVectorToIndex(Point* p) const
  {
    *p *= _inverseDelta;
  }

  //! Calculate the index range that contains all of the indices in the box.
  /*! The window is first converted to index coordinates and then rounded. */
  void
  computeRange(const BBox& box, Range* range) const
  {
    IndexList lower, upper;

    Point p = box.lower;
    convertLocationToIndex(&p);
    for (std::size_t i = 0; i != lower.size(); ++i) {
      // Round up to the closed lower bound.
      // Use only points in the grid range.
      lower[i] = std::max(Index(std::ceil(p[i])), Index(0));
    }

    p = box.upper;
    convertLocationToIndex(&p);
    for (std::size_t i = 0; i != upper.size(); ++i) {
      // Round up to get an open upper bound.
      // Use only points in the grid range.
      // Indicate an empty range by setting the upper bound to the lower
      // bound.
      upper[i] = std::max(std::min(Index(std::ceil(p[i])),
                                   Index(_extents[i])),
                          lower[i]);
    }

    range->initialize(lower, upper);
  }

  //! Convert the Cartesian coordinates of the bounding box to a grid index coordinates.
  void
  convertBBoxLocationsToIndices(BBox* box) const
  {
    convertLocationToIndex(&box->lower);
    convertLocationToIndex(&box->upper);
  }

  //! Convert the grid index coordinates of a bounding box to Cartesian coordinates.
  void
  convertBBoxIndicesToLocations(BBox* box) const
  {
    convertIndexToLocation(&box->lower);
    convertIndexToLocation(&box->upper);
  }

  //! Convert the grid index coordinates of the first bounding box to Cartesian coordinates in the second.
  template<typename _Index>
  void
  convertBBoxIndicesToLocations(const geom::BBox<_Index, N>& indexBox,
                                BBox* cartesianBox) const
  {
    // Lower corner.
    Point p;
    // First convert to the number type.
    for (std::size_t i = 0; i != p.size(); ++i) {
      p[i] = indexBox.lower[i];
    }
    convertIndexToLocation(&p);
    cartesianBox->lower = p;
    // Upper corner.
    for (std::size_t i = 0; i != p.size(); ++i) {
      p[i] = indexBox.upper[i];
    }
    convertIndexToLocation(&p);
    cartesianBox->upper = p;
  }

  //! Convert a set of Cartesian coordinates to grid index coordinates.
  template<typename OutputIterator>
  void
  convertLocationsToIndices(OutputIterator begin, OutputIterator end) const
  {
    for (; begin != end; ++begin) {
      convertLocationToIndex(&*begin);
    }
  }

  //! Convert a set of grid index coordinates to Cartesian coordinates.
  template<typename OutputIterator>
  void
  convertIndicesToLocations(OutputIterator begin, OutputIterator end) const
  {
    for (; begin != end; ++begin) {
      convertIndexToLocation(&*begin);
    }
  }

  //! Convert a set of vectors in Cartesian coordinates to index coordinates.
  template<typename OutputIterator>
  void
  convertVectorsToIndices(OutputIterator begin, OutputIterator end) const
  {
    for (; begin != end; ++begin) {
      convertVectorToIndex(&*begin);
    }
  }

  // @}
};


//
// Equality operators.
//


//! Return true if the true RegularGrid's are equal.
/*! \relates RegularGrid */
template<std::size_t N, typename T>
bool
operator==(const RegularGrid<N, T>& a, const RegularGrid<N, T>& b);


//! Return true if the true RegularGrid's are not equal.
/*! \relates RegularGrid */
template<std::size_t N, typename T>
inline
bool
operator!=(const RegularGrid<N, T>& a, const RegularGrid<N, T>& b)
{
  return !(a == b);
}


//
// File I/O
//


//! Write to a file stream.
/*! \relates RegularGrid */
template<std::size_t N, typename T>
std::ostream&
operator<<(std::ostream& out, const RegularGrid<N, T>& grid);


//! Read from a file stream.
/*! \relates RegularGrid */
template<std::size_t N, typename T>
std::istream&
operator>>(std::istream& in, RegularGrid<N, T>& grid);


} // namespace geom
}

#define __geom_RegularGrid_ipp__
#include "stlib/geom/grid/RegularGrid.ipp"
#undef __geom_RegularGrid_ipp__

#endif
