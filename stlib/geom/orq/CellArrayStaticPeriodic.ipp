// -*- C++ -*-

#if !defined(__geom_orq_CellArrayStaticPeriodic_ipp__)
#error This file is an implementation detail of the class CellArrayStaticPeriodic.
#endif

namespace stlib
{
namespace geom
{


template<std::size_t N, typename _Location>
inline
CellArrayStaticPeriodic<N, _Location>::
CellArrayStaticPeriodic(const BBox& domain, Record first, Record last) :
  // Instantiate the location functor.
  _location(),
  // The class for computing periodic distances.
  _distance(domain),
  // For the cell array, start with invalid values.
  _extents(),
  _strides(),
  _inverseCellSizes(),
  _cellArray()
{
  set(first, last);
}


template<std::size_t N, typename _Location>
inline
void
CellArrayStaticPeriodic<N, _Location>::
set(Record first, Record last)
{
  // Clear the records.
  clear();
  // Compute the extents for the cell array.
  // Note that we deal with the trivial (empty) case.
  computeExtentsAndSizes(std::max(std::distance(first, last),
                                  std::ptrdiff_t(1)));
  // Determine where each record should be placed and the sizes
  // (number of records) for each cell.
  std::vector<std::size_t> indices(std::distance(first, last));
  std::vector<std::size_t> sizes(ext::product(_extents), std::size_t(0));
  Record record = first;
  for (std::size_t i = 0; i != indices.size(); ++i) {
    indices[i] = cellIndex(periodicIndices(plainIndices
                                           (_location(record++))));
    ++sizes[indices[i]];
  }
  // Allocate the static array of arrays that represent the cell array.
  _cellArray.rebuild(sizes.begin(), sizes.end());
  // Copy the records into the array.
  std::vector<typename container::StaticArrayOfArrays<Record>::iterator>
  positions(sizes.size());
  for (std::size_t i = 0; i != positions.size(); ++i) {
    positions[i] = _cellArray.begin(i);
  }
  record = first;
  for (std::size_t i = 0; i != indices.size(); ++i) {
    *positions[indices[i]]++ = record++;
  }
}


template<std::size_t N, typename _Location>
inline
void
CellArrayStaticPeriodic<N, _Location>::
computeExtentsAndSizes(const std::size_t suggestedNumberOfCells)
{
  assert(suggestedNumberOfCells > 0);

  // Work from the the least to greatest Cartesian extent to compute the
  // grid extents.
  Point ext = domain().upper - domain().lower;
  std::array<std::size_t, N> order;
  ads::computeOrder(ext.begin(), ext.end(), order.begin());
  for (std::size_t i = 0; i != N; ++i) {
    // Normalize the domain to suggestedNumberOfCells content.
    Float content = ext::product(ext);
    assert(content != 0);
    const Float factor = std::pow(suggestedNumberOfCells / content,
                                  Float(1.0 / (N - i)));
    for (std::size_t j = i; j != N; ++j) {
      ext[order[j]] *= factor;
    }
    // The current index;
    const std::size_t n = order[i];
    // Add 0.5 and truncate to round to the nearest integer.
    ext[n] = _extents[n] = std::max(std::size_t(1),
                                    std::size_t(ext[n] + 0.5));
  }

  // Compute the strides from the extents.
  _strides[0] = 1;
  for (std::size_t n = 1; n != N; ++n) {
    _strides[n] = _strides[n - 1] * _extents[n - 1];
  }

  // From the domain and the cell array extents, compute the cell size.
  for (std::size_t n = 0; n != N; ++n) {
    const Float d = domain().upper[n] - domain().lower[n];
    if (d == 0 || _extents[n] == 1) {
      // The cell covers the entire domain.
      _inverseCellSizes[n] = 0;
    }
    else {
      _inverseCellSizes[n] = _extents[n] / d;
    }
  }
}


template<std::size_t N, typename _Location>
inline
typename CellArrayStaticPeriodic<N, _Location>::IndexList
CellArrayStaticPeriodic<N, _Location>::
plainIndices(const Point& location) const
{
  IndexList indices;
  for (std::size_t i = 0; i != N; ++i) {
    indices[i] = Index(std::floor((location[i] - domain().lower[i]) *
                                  _inverseCellSizes[i]));
  }
  return indices;
}


template<std::size_t N, typename _Location>
inline
typename CellArrayStaticPeriodic<N, _Location>::IndexList
CellArrayStaticPeriodic<N, _Location>::
periodicIndices(IndexList indices) const
{
  for (std::size_t i = 0; i != indices.size(); ++i) {
    indices[i] = (indices[i] % _extents[i] + _extents[i]) % _extents[i];
  }
  return indices;
}


template<std::size_t N, typename _Location>
inline
std::size_t
CellArrayStaticPeriodic<N, _Location>::
cellIndex(const IndexList& indices) const
{
#ifdef STLIB_DEBUG
  for (std::size_t i = 0; i != indices.size(); ++i) {
    assert(0 <= indices[i] && indices[i] < _extents[i]);
  }
#endif
  return ext::dot(indices, _strides);
}


template<std::size_t N, typename _Location>
template<class _OutputIterator>
inline
std::size_t
CellArrayStaticPeriodic<N, _Location>::
neighborQuery(_OutputIterator iter, const Ball& ball) const
{
  // Plain indices for the (closed) lower and (open) upper corners of the
  // box that bounds the ball.
  IndexList lower = plainIndices(ball.center - ball.radius);
  IndexList upper = plainIndices(ball.center + ball.radius) + Index(1);
  // Limit the extents so that cells are not checked multiple times.
  for (std::size_t i = 0; i != N; ++i) {
    if (upper[i] - lower[i] > _extents[i]) {
      upper[i] = lower[i] + _extents[i];
    }
  }
  // Define the cell index range, using plain indices.
  container::MultiIndexRange<N> indexRange(lower, upper);

  //
  // Iterate over the cells in the index window.
  //
  // The number of records in the window.
  std::size_t count = 0;
  const Float squaredDistance = ball.radius * ball.radius;
  typename container::StaticArrayOfArrays<Record>::const_iterator r, rEnd;
  container::MultiIndexRangeIterator<N> indicesEnd =
    container::MultiIndexRangeIterator<N>::end(indexRange);
  for (container::MultiIndexRangeIterator<N> indices =
         container::MultiIndexRangeIterator<N>::begin(indexRange);
       indices != indicesEnd; ++indices) {
    const std::size_t index = cellIndex(periodicIndices(*indices));
    rEnd = _cellArray.end(index);
    for (r = _cellArray.begin(index); r != rEnd; ++r) {
      // If the record is in the ball.
      if (_distance.squaredDistance(ball.center, _location(*r)) <=
          squaredDistance) {
        *iter = *r;
        ++iter;
        ++count;
      }
    }
  }

  return count;
}

//
// File I/O
//

template<std::size_t N, typename _Location>
inline
void
CellArrayStaticPeriodic<N, _Location>::
put(std::ostream& out) const
{
  // Write the multi-keys.
  for (typename container::StaticArrayOfArrays<Record>::const_iterator
       i = _cellArray.begin(); i != _cellArray.end(); ++i) {
    out << _location(*i) << '\n';
  }
}

} // namespace geom
}
