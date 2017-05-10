// -*- C++ -*-

/*!
  \file CellArrayNeighbors.h
  \brief A class for computing neighbors within a specified radius.
*/

#if !defined(__geom_CellArrayNeighbors_h__)
#define __geom_CellArrayNeighbors_h__

#include "stlib/geom/kernel/BBox.h"
#include "stlib/container/SimpleMultiArray.h"
#include "stlib/container/SimpleMultiIndexRangeIterator.h"
#include "stlib/ads/functor/Dereference.h"
#include "stlib/ads/algorithm/sort.h"

#include <cstring>

namespace stlib
{
namespace geom
{

//! A class for computing neighbors within a specified radius.
/*!
  \param _Float The floating-point number type may be either \c float
  or \c double.
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
  without any arguments. One must pass a location functor
  only if it does not have a default constructor.
  \code
  const double searchRadius = 1;
  geom::CellArrayNeighbors<double, 3, Node*, Location> neighborSearch;
  \endcode

  \par Neighbor queries.
  First use the \c initialize() member function to register the records.
  These will be sorted and stored in the cell array. One must re-initialize
  whenever the locations of the records are modified.
  Next use the \c neighborsQuery() member function to find the neighbors
  of specified points. Specifically, it finds the records within a ball
  with given center and radius.
  \code
  std::vector<Node> nodes;
  std::vector<geom::Ball<double, 3> > balls;
  ...
  neighborSearch.initialize(&nodes[0], &nodes[0] + nodes.size());
  std::vector<Node*> neighbors;
  // For each query ball.
  for (std::size_t i = 0; i != balls.size(); ++i) {
     neighborSearch.neighborQuery(balls[i].center, balls[i].radius, &neighbors);
     // For each neighbor of this point.
     for (std::size_t j = 0; j != neighbors.size(); ++j) {
        // Do something with the neighbor.
        foo(neighbors[j]);
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

  \par 3-D Performance.
  We perform neighbor queries for a set of records whose coordinates have
  various distributions (we consider uniform, Gaussian, and exponential).
  The random deviates are scaled by the cube root of the number of
  records so that number of neighbor for a given search radius is
  roughly constant. We perform a
  query around the center of each record and choose a
  search radius to yield roughly 10 neighbors per query for the
  case of one million records. (The radii are 1.297, 4.579, and 2.608
  for the uniform, Gaussian, and exponential distributions, respectively.)
  Below we show the average number of neighbors for each as we vary the number
  of records.

  \par
  <table>
  <tr>
  <th> Records
  <th> Uniform
  <th> Gaussian
  <th> Exponential
  <tr>
  <td> 1,000
  <td> 8.6
  <td> 9.5
  <td> 8.4
  <tr>
  <td> 10,000
  <td> 9.5
  <td> 10.1
  <td> 9.0
  <tr>
  <td> 100,000
  <td> 9.9
  <td> 10.0
  <td> 9.7
  <tr>
  <td> 1,000,000
  <td> 10.0
  <td> 10.0
  <td> 10.0
  <tr>
  <td> 10,000,000
  <td> 10.1
  <td> 10.0
  <td> 10.2
  </table>

  \par
  For the uniform distribution, the points are uniformly distributed in a
  unit box, and then scaled by a constant factor. Obviously, the
  uniform distribution is the easiest of the three problems. For the
  Gaussian distribution, the points are densely clustered near the origin
  and spread widely at the edges of the domain. Because of its long
  tail, the exponential distribution is the most challenging of the
  three problems. To give an indication of the spread of the points,
  we compare the mean length of the sides of the bounding box containing
  the records in the table below.

  \par
  <table>
  <tr>
  <th> Records
  <th> Uniform
  <th> Gaussian
  <th> Exponential
  <tr>
  <td> 1,000
  <td> 9.99
  <td> 68.8
  <td> 81.2
  <tr>
  <td> 10,000
  <td> 21.5
  <td> 161
  <td> 206
  <tr>
  <td> 100,000
  <td> 46.4
  <td> 402
  <td> 572
  <tr>
  <td> 1,000,000
  <td> 100
  <td> 973
  <td> 1480
  <tr>
  <td> 10,000,000
  <td> 215
  <td> 2250
  <td> 3780
  </table>

  \par
  Below we show performance results for the initialization step. Times
  are given in seconds. In the vertical direction we vary the number
  of records. In the horizontal direction we first vary the point
  distribution and then vary the ordering of the points.
  Note that both the initialization and the query times
  depend upon this ordering. We consider
  three different alternatives: random, sorted by z coordinate,
  and ordered with a Morton spatial index. For the last option,
  we use the \c geom::SpatialIndexMortonUniform class and set the maximum
  block size to the query radius.
  The test is
  conducted on a MacBook Pro with a 2.8 GHz Intel Core 2 Duo processor
  with 8 GB of 1067 MHz DDR3 RAM.

  \par
  <table>
  <tr>
  <th>
  <th>
  <th> Uniform
  <th>
  <th>
  <th>
  <th> Gaussian
  <th>
  <th>
  <th>
  <th> Exponential
  <th>
  <th>
  <tr>
  <th> Records
  <th>
  <th> Random
  <th> Sort by Z
  <th> Morton
  <th>
  <th> Random
  <th> Sort by Z
  <th> Morton
  <th>
  <th> Random
  <th> Sort by Z
  <th> Morton
  <tr>
  <td> 1,000
  <td>
  <td> 0.000164
  <td> 0.000173
  <td> 0.000121
  <td>
  <td> 0.000154
  <td> 0.000225
  <td> 0.000125
  <td>
  <td> 0.000181
  <td> 0.000175
  <td> 0.000125
  <tr>
  <td> 10,000
  <td>
  <td> 0.00192
  <td> 0.00195
  <td> 0.00116
  <td>
  <td> 0.00196
  <td> 0.00181
  <td> 0.00117
  <td>
  <td> 0.00190
  <td> 0.00181
  <td> 0.00137
  <tr>
  <td> 100,000
  <td>
  <td> 0.0185
  <td> 0.0188
  <td> 0.0151
  <td>
  <td> 0.0178
  <td> 0.0173
  <td> 0.0143
  <td>
  <td> 0.0177
  <td> 0.0178
  <td> 0.0148
  <tr>
  <td> 1,000,000
  <td>
  <td> 0.294
  <td> 0.185
  <td> 0.175
  <td>
  <td> 0.201
  <td> 0.179
  <td> 0.163
  <td>
  <td> 0.181
  <td> 0.176
  <td> 0.158
  <tr>
  <td> 10,000,000
  <td>
  <td> 3.91
  <td> 2.15
  <td> 2.09
  <td>
  <td> 3.04
  <td> 2.08
  <td> 1.98
  <td>
  <td> 2.27
  <td> 2.03
  <td> 1.95
  </table>

  \par
  Above we see that the cost of initialization is nearly linear in the
  number of records. While the asymptotic computational complexity is
  linear, the actual performance is affected by the memory hierarchy.
  Both the distribution of the points and the ordering of the points
  have a modest effect on performance. To minimize the initialization time,
  it is useful to use the Morton ordering or to sort by the z-coordinate.

  \par
  Next we consider the performance of the neighbor queries, which we perform
  on each record, in order. In the table below, the average
  time per reported neighbor is given in nanoseconds.

  \par
  <table>
  <tr>
  <th>
  <th>
  <th> Uniform
  <th>
  <th>
  <th>
  <th> Gaussian
  <th>
  <th>
  <th>
  <th> Exponential
  <th>
  <th>
  <tr>
  <th> Records
  <th>
  <th> Random
  <th> Sort by Z
  <th> Morton
  <th>
  <th> Random
  <th> Sort by Z
  <th> Morton
  <th>
  <th> Random
  <th> Sort by Z
  <th> Morton
  <tr>
  <td> 1,000
  <td>
  <td> 78.0
  <td> 75.4
  <td> 74.4
  <td>
  <td> 72.6
  <td> 70.6
  <td> 70.0
  <td>
  <td> 95.5
  <td> 86.6
  <td> 83.2
  <tr>
  <td> 10,000
  <td>
  <td> 88.8
  <td> 71.1
  <td> 68.6
  <td>
  <td> 75.7
  <td> 66.8
  <td> 65.4
  <td>
  <td> 126
  <td> 123
  <td> 109
  <tr>
  <td> 100,000
  <td>
  <td> 76.8
  <td> 71.4
  <td> 59.2
  <td>
  <td> 74.0
  <td> 69.3
  <td> 62.6
  <td>
  <td> 198
  <td> 191
  <td> 181
  <tr>
  <td> 1,000,000
  <td>
  <td> 277
  <td> 72.7
  <td> 60.5
  <td>
  <td> 178
  <td> 77.9
  <td> 68.8
  <td>
  <td> 546
  <td> 302
  <td> 278
  <tr>
  <td> 10,000,000
  <td>
  <td> 357
  <td> 119
  <td> 61.5
  <td>
  <td> 239
  <td> 127
  <td> 73.0
  <td>
  <td> 984
  <td> 765
  <td> 430
  </table>

  \par
  Note that for the uniform and Gaussian distribution, in each column the
  times are roughly constant until cache misses degrade the performance.
  For the exponential distribution, the cost increases at an accelerating
  rate as the number of records increases. Besides the cache effects,
  the distribution of points becomes more inhomogeneous as the
  the number of records increases.

  \par
  As one would expect, the cost of neighbor queries increases with
  increasing inhomogeneity in the distribution of records. This is because
  the size of the cells increases. If the cells in the array are much
  larger than the query windows, then many candidate records must be
  examined in order to determine the records that lie in the ball.
  If one uses the Morton ordering, the effect is modest for
  the Gaussian distribution. However, it becomes
  pronounced for large numbers of records with the exponential
  distribution.

  \par
  We see that for a small number of records, the ordering of the queries
  has little effect. However, for larger data sets ordering the queries
  reduces the cache misses and significantly improves performance.
  Thus, it is recomended that the records be ordered before constructing
  the neigbor query class, and that the order of the queries match the
  storage order of the records. Simply sorting the records by the z
  coordinate is an effective approach, but using the more sophisticated
  Morton ordering is better.

  \par Threading.
  The initialize() function must be called in a serial block, or within a
  single thread. However, the neighborQuery() functions are thread-safe.
  After initialization, one may perform neighbor queries in a threaded
  block. If the records have been ordered to improve cache utilization,
  then it is best if each thread is responsible for a contiguous
  block of records.
*/
template<typename _Float,
         std::size_t _D,
         typename _Record,
         typename _Location = ads::Dereference<_Record> >
class CellArrayNeighbors
{
  //
  // Constants.
  //
public:

  //! The space dimension.
  BOOST_STATIC_CONSTEXPR std::size_t D = _D;

  //
  // Types.
  //
public:

  //! The floating-point number type.
  typedef _Float Float;
  //! A pointer to the record type.
  typedef _Record Record;
  //! A Cartesian point.
  typedef std::array<Float, D> Point;

protected:

  //! Bounding box.
  typedef geom::BBox<Float, D> BBox;
  //! A multi-index.
  typedef typename container::SimpleMultiArray<int, D>::IndexList IndexList;
  //! A single index.
  typedef typename container::SimpleMultiArray<int, D>::Index Index;

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

private:

  //! The functor for computing record locations.
  _Location _location;
  //! The records along with the cell indices and locations.
  std::vector<RecLoc> _recordData;
  //! The array of cells.
  /*! The array is padded by one empty slice in the first dimension.
    This makes it easier to iterate over a range of cells. */
  container::SimpleMultiArray<ConstIterator, D> _cellArray;
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

  // Copy constructor not implemented. _cellArray has iterators to _recordData
  // so the synthesized copy constructor would not work.
  CellArrayNeighbors(const CellArrayNeighbors&);

  // Assignment operator not implemented.
  CellArrayNeighbors&
  operator=(const CellArrayNeighbors&);

  //--------------------------------------------------------------------------
  //! \name Constructors.
  // @{
public:

  //! Construct from the location functor.
  CellArrayNeighbors(const _Location& location = _Location()) :
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

  //! Initialize with the given sequence of records.
  template<typename _InputIterator>
  void
  initialize(_InputIterator begin, _InputIterator end);

  //! Find the records that are in the specified ball.
  template<typename _OutputIterator>
  void
  neighborQuery(const Point& center, Float radius, _OutputIterator neighbors)
  const;

  //! Find the records that are in the specified ball.
  /*! Store the neighbors in the supplied container, which will first be
    cleared. */
  void
  neighborQuery(const Point& center, Float radius,
                std::vector<Record>* neighbors) const
  {
    neighbors->clear();
    neighborQuery(center, radius, std::back_inserter(*neighbors));
  }

protected:

  //! Convert a location to a valid cell array multi-index.
  IndexList
  locationToIndices(const Point& x) const;

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

#define __geom_CellArrayNeighbors_ipp__
#include "stlib/geom/orq/CellArrayNeighbors.ipp"
#undef __geom_CellArrayNeighbors_ipp__

// 3-D specialization.
#define __geom_CellArrayNeighbors3_ipp__
#include "stlib/geom/orq/CellArrayNeighbors3.ipp"
#undef __geom_CellArrayNeighbors3_ipp__

#endif
