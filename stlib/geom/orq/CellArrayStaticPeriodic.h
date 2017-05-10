// -*- C++ -*-

/*!
  \file CellArrayStaticPeriodic.h
  \brief A class for a static cell array in N-D on a periodic domain.
*/

#if !defined(__geom_orq_CellArrayStaticPeriodic_h__)
#define __geom_orq_CellArrayStaticPeriodic_h__

#include "stlib/geom/orq/DistancePeriodic.h"
#include "stlib/geom/kernel/Ball.h"
#include "stlib/ads/algorithm/sort.h"
#include "stlib/container/MultiIndexRangeIterator.h"
#include "stlib/container/StaticArrayOfArrays.h"

namespace stlib
{
namespace geom
{

//! A static cell array in N-D on a periodic domain.
/*!
  This class uses DistancePeriodic to compute distance between points.
*/
template<std::size_t N, typename _Location>
class CellArrayStaticPeriodic
{
  //
  // Types.
  //
public:
  //! The multi-key accessor.
  typedef _Location Location;
  //! The record type.
  typedef typename Location::argument_type Record;
  //! A Cartesian point. Note that \c result_type could be a reference or const reference.
  typedef typename
  std::remove_const<typename std::remove_reference<typename Location::result_type>::type>::type
  Point;
  //! The key type.
  typedef typename Point::value_type Float;
  //! Bounding box.
  typedef geom::BBox<Float, N> BBox;
  //! A ball.
  typedef geom::Ball<Float, N> Ball;

private:

  //! An index is a signed integer.
  typedef std::ptrdiff_t Index;
  //! A multi-index.
  typedef std::array<Index, N> IndexList;

  //
  // Data
  //
private:

  //! The multi-key accessor.
  Location _location;
  //! The structure for computing distance on a periodic domain.
  DistancePeriodic<Float, N> _distance;
  //! The number of cells in each dimension.
  IndexList _extents;
  //! The strides are used for array indexing.
  IndexList _strides;
  //! The inverse cell sizes.
  Point _inverseCellSizes;
  //! The array of cells.
  container::StaticArrayOfArrays<Record> _cellArray;

  //--------------------------------------------------------------------------
  //! \name Constructors.
  // @{
public:

  //! Construct from the domain and the records.
  /*!
    The number of cells will be approximately the number of records.

    \param domain The Cartesian domain that contains the records.
    \param first The first record.
    \param last The last record.

    \note This function assumes that the records are iterators.
  */
  CellArrayStaticPeriodic(const BBox& domain, Record first, Record last);

  // @}
  //--------------------------------------------------------------------------
  //! \name set/clear records.
  // @{
public:

  //! Set the records.
  /*!
    \param first The first record.
    \param last The last record.
  */
  void
  set(Record first, Record last);

  //! Clear all records.
  void
  clear()
  {
    std::fill(_extents.begin(), _extents.end(), 0);
    _cellArray.clear();
  }

private:

  //! Compute the array extents and the sizes for the cells.
  void
  computeExtentsAndSizes(const std::size_t suggestedNumberOfCells);

  //! Convert the location to plain indices (which may lie outside the array).
  IndexList
  plainIndices(const Point& location) const;

  //! Convert plain array indices to periodic indices.
  IndexList
  periodicIndices(IndexList indices) const;

  //! Convert the index list to a cell index.
  std::size_t
  cellIndex(const IndexList& indices) const;

  // @}
  //--------------------------------------------------------------------------
  //! \name Accessors.
  // @{

  //! Return the number of records.
  std::size_t
  size() const
  {
    return _cellArray.size();
  }

  //! Return true if there are no records.
  bool
  empty() const
  {
    return size() == 0;
  }

  //! Return the domain.
  const BBox&
  domain() const
  {
    return _distance.domain();
  }

  // @}
  //--------------------------------------------------------------------------
  //! \name Queries.
  // @{
public:

  //! Get the records in the window. Return the # of records inside.
  template<typename _OutputIterator>
  std::size_t
  neighborQuery(_OutputIterator iter, const Ball& ball) const;

  // @}
  //--------------------------------------------------------------------------
  //! \name File I/O.
  // @{
public:

  //! Print the records.
  void
  put(std::ostream& out) const;

  // @}
};

//
// File I/O
//

//! Write to a file stream.
/*! \relates CellArrayStaticPeriodic */
template<std::size_t N, typename _Location>
inline
std::ostream&
operator<<(std::ostream& out, const CellArrayStaticPeriodic<N, _Location>& x)
{
  x.put(out);
  return out;
}

} // namespace geom
}

#define __geom_orq_CellArrayStaticPeriodic_ipp__
#include "stlib/geom/orq/CellArrayStaticPeriodic.ipp"
#undef __geom_orq_CellArrayStaticPeriodic_ipp__

#endif
