// -*- C++ -*-

/*!
  \file CellArrayAllNeighbors.h
  \brief A class for computing all neighbors of the input records.
*/

#if !defined(__geom_CellArrayAllNeighbors_h__)
#define __geom_CellArrayAllNeighbors_h__

// Not a base class, just used for types.
#include "stlib/geom/orq/ORQ.h"

#include "stlib/container/SimpleMultiArray.h"
#include "stlib/container/SimpleMultiIndexRangeIterator.h"
#include "stlib/ads/functor/Dereference.h"
#include "stlib/ads/algorithm/sort.h"

#include <cstring>

namespace stlib
{
namespace geom
{

//! A class for computing all neighbors of the input records.
/*!
  \param _D The space dimension.
  \param _Record The record type, which is most likely a pointer to a class
  or an iterator into a container.
  \param _Location A functor that takes the record type as an argument
  and returns the location for the record.

  \par The Location functor.
  Suppose that one has the following Node structure for representing a node
  in a 3-D simulation. The \c coords member points to some externally allocated
  memory.
  \code
  struct Node {
     double* coords;
  };
  \endcode
  Next suppose that the record type is a pointer to Node. Below is a functor
  that converts a record to a Cartesian point (\c std::array<double,3>).
  \code
  struct Location :
     public std::unary_function<Node*, std::array<double,3> > {
     result_type
     operator()(argument_type r) {
        result_type location = {{r->coords[0], r->coords[1], r->coords[2]}};
        return location;
     }
  };
  \endcode

  \par Constructor.
  In most circumstances, one constructs the neighbors search data structure
  simply by specifying the search radius. One must pass a location functor
  only if it does not have a default constructor.
  \code
  const double searchRadius = 1;
  geom::CellArrayAllNeighbors<3, Node*, Location> neighborSearch(searchRadius);
  \endcode

  \par Neighbor queries.
  Use the \c allNeighbors() member function to calculate the neighbors for
  all of the records. The neighbor information will be stored in the
  three data members: \c records, \c packedNeighbors, and \c neighborDelimiters.
  \c records is a re-ordering of the input records.
  The re-ordering is done to improve cache utilization when performing the
  neighbor queries.
  \c packedNeighbors is a vector of the neighbors for all of the records.
  neighborDelimiters defines the ranges for the neighbors of each record.
  Consider the nth record in the \c records data member.
  Its first neighbor is \c packedNeighbors[neighborDelimiters[n]].
  One past the last neighbor is \c packedNeighbors[neighborDelimiters[n+1]].
  \code
  std::vector<Node> nodes;
  ...
  neighborSearch.allNeighbors(&nodes[0], &nodes[0] + nodes.size());
  // For each record.
  for (std::size_t i = 0; i != neighborSearch.records.size(); ++i) {
     Node* record = neighborSearch.records[i];
     // For each neighbor of this record.
     for (std::size_t j = neighborSearch.neighborDelimiters[i];
          j != neighborSearch.neighborDelimiters[i+1]; ++j) {
        Node* neighbor = neighborSearch.packedNeighbors[j];
        // Do something with the node and its neighbor.
        ...
     }
  }
  \endcode

  \par Neighbor queries and %ORQ data structures.
  One performs neighbor queries by first bounding the search ball with an
  axis-oriented bounding box. One then uses an orthogonal range query
  (%ORQ) data structure to find all of the records in the bounding box.
  Octrees, K-d trees, and cell arrays are common %ORQ data structures.
  The tree data structures are heirarchical; the leaves store small sets
  of records with a fixed (small) maximum size. A dense cell array covers
  the bounding box of the records with a uniform array of cells. For most
  record distributions, cell arrays are more efficient than tree data
  structures. One can directly access and iterate over the cells that
  intersect the query window.

  \par Efficient cell representation.
  The simplest way to store a cell array would be use an array of
  variabled-sized containers. For example, a multi-dimensional array
  of \c std::vector's.
  \code
   container::SimpleMultiArray<std::vector<Record>, D> cellArray;
  \endcode
  This would be very inefficient both in terms of storage and cache
  utilization. We store pairs of the records and their locations in
  a packed vector. (We store the location to avoid the cost of
  recomputing it from the record.) The record/location pairs are
  sorted according to their container index in the multi-array.
  (Note that the sorting may be done in linear time.) The cell array
  is simply a multi-array of iterators into the vector of
  record/location pairs. The iterator for a given cell points to the
  first record in that cell. One uses the following cell to obtain
  the end iterator.

  \par Cell size.
  Choosing a good cell size is important for the performance of %ORQ's
  with cell arrays. If the cells are much larger than the query windows
  then one will have to examine many more candidate records than the
  number of neighbors for any particular query. If the cells are much
  smaller than the query window, then the cost of iterating over cells
  may dominate. If the search radius is fixed, a common approach is
  to set the cell lengths equal to the search radius. However, this
  is problematic because, depending on the distribution of the record
  locations, there may be many more cells than records. The solution
  is to set the number of cells to be approximately equal to the number
  of records and then compute appropriate cell dimensions based on
  that constraint.

  \par Re-ordering records.
  The records are re-ordered to improve data locality (and hence cache
  utilization) when performing the neighbor queries. The neighbor queries
  are performed in the same order as the records are stored in the
  cell array. This significantly improves performance. For a test with
  one million records, ordering the neighbor queries in this manner reduces
  the execution time by a factor of four.

  \par Performance.
  We perform neighbor queries for a set of records uniformly distributed
  in a cube whose volume is equal to the number of records. We choose a
  search radius of 1.4 to yield roughly 10 neighbors per record. Below
  we show performance results as we vary the number of records. The test
  is conducted on a MacBook Pro with a 2.8 GHz Intel Core 2 Duo processor
  with 4 GB of 1067 MHz DDR3 RAM.
  <table>
  <tr>
  <th> Records
  <th> Average # of neighbors
  <th> Time for all neighbors
  <th> Time per neighbor
  <tr>
  <td> 1,000
  <td> 9.7
  <td> 0.92 milliseconds
  <td> 95 nanoseconds
  <tr>
  <td> 10,000
  <td> 10.7
  <td> 7.9 milliseconds
  <td> 74 nanoseconds
  <tr>
  <td> 100,000
  <td> 11.1
  <td> 91 milliseconds
  <td> 82 nanoseconds
  <tr>
  <td> 1,000,000
  <td> 11.3
  <td> 1.1 seconds
  <td> 95 nanoseconds
  </table>
*/
template<std::size_t _D,
         typename _Record,
         typename _Location = ads::Dereference<_Record> >
class CellArrayAllNeighbors
{
  //
  // Types.
  //
public:

  //! A pointer to the record type.
  typedef _Record Record;
  //! A Cartesian point.
  typedef typename Orq<_D, _Location>::Point Point;
  //! The floating-point number type.
  typedef typename Orq<_D, _Location>::Float Float;

protected:

  //! Bounding box.
  typedef typename Orq<_D, _Location>::BBox BBox;
  //! A multi-index.
  typedef typename container::SimpleMultiArray<int, _D>::IndexList IndexList;
  //! A single index.
  typedef typename container::SimpleMultiArray<int, _D>::Index Index;

  //
  // Nested classes.
  //

private:

  struct RecLoc {
    Record record;
    Point location;
  };

  typedef typename std::vector<RecLoc>::const_iterator ConstIterator;

  //
  // Data
  //

public:

  //! The search radius for finding neighbors.
  const Float searchRadius;
  //! The sequence of records for which the neighbors were computed.
  /*! Note that the order may not be the same as for the input sequence.
    They are re-ordered to improve cache utilization. */
  std::vector<Record> records;
  //! The packed sequence of neighbors for the records.
  std::vector<Record> packedNeighbors;
  //! The delimiters for the neighbor lists.
  /*! The first neighbor for the nth record is
    \c packedNeighbors[neighborDelimiters[n]]. One past the last neighbor is
    \c packedNeighbors[neighborDelimiters[n+1]]. */
  std::vector<std::size_t> neighborDelimiters;

private:

  //! The functor for computing record locations.
  _Location _location;
  //! The records along with the cell indices and locations.
  std::vector<RecLoc> _recordData;
  //! The array of cells.
  /*! The array is padded by one empty slice in the first dimension.
    This makes it easier to iterate over a range of cells. */
  container::SimpleMultiArray<ConstIterator, _D> _cellArray;
  //! The Cartesian location of the lower corner of the cell array.
  Point _lowerCorner;
  //! The inverse cell lengths.
  /*! This is used for converting locations to cell multi-indices. */
  Point _inverseCellLengths;

  // Scratch data for cellSort().
  std::vector<Index> _cellIndices;
  std::vector<std::size_t> _cellCounts;
  std::vector<RecLoc> _recordDataCopy;

private:

  //
  // Not implemented
  //

  // Copy constructor not implemented
  CellArrayAllNeighbors(const CellArrayAllNeighbors&);

  // Assignment operator not implemented
  CellArrayAllNeighbors&
  operator=(const CellArrayAllNeighbors&);

  //--------------------------------------------------------------------------
  //! \name Constructors.
  // @{
public:

  //! Construct from the search radius.
  CellArrayAllNeighbors(const Float searchRadius_,
                        const _Location& location = _Location()) :
    searchRadius(searchRadius_),
    records(),
    packedNeighbors(),
    neighborDelimiters(),
    _location(location),
    _recordData(),
    _cellArray(),
    // Fill with invalid values.
    _lowerCorner(ext::filled_array<Point>
                 (std::numeric_limits<Float>::quiet_NaN())),
    _inverseCellLengths(ext::filled_array<Point>
                        (std::numeric_limits<Float>::quiet_NaN()))
  {
  }

  // @}
  //--------------------------------------------------------------------------
  //! \name Window Queries.
  // @{
public:

  //! Compute the neighbors for each record.
  /*! Store the results in the following member data: records,
    packedNeighbors, and neighborDelimiters.
    \note The _InputIterator type must be assignable to a Record.
  */
  template<typename _InputIterator>
  void
  allNeighbors(_InputIterator begin, _InputIterator end);

protected:

  //! Record the neighbors for the specified record in packedRecords.
  void
  recordNeighbors(const std::size_t index);

  //! Convert a location to a valid cell array multi-index.
  IndexList
  locationToIndices(const Point& x);

  //! Convert a location to a container index.
  Index
  containerIndex(const Point& x);

  //! Sort _recordData by cell container indices.
  void
  cellSort();

  //! Compute the array extents and the sizes for the cells.
  IndexList
  computeExtentsAndSizes(std::size_t numberOfCells, const BBox& domain);

  // @}
};

} // namespace geom
}

#define __geom_CellArrayAllNeighbors_ipp__
#include "stlib/geom/orq/CellArrayAllNeighbors.ipp"
#undef __geom_CellArrayAllNeighbors_ipp__

#endif
