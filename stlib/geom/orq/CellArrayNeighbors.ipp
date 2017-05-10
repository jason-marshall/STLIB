// -*- C++ -*-

#if !defined(__geom_CellArrayNeighbors_ipp__)
#error This file is an implementation detail of the class CellArrayNeighbors.
#endif

namespace stlib
{
namespace geom
{


template<typename _Float, std::size_t _D, typename _Record, typename _Location>
template<typename _InputIterator>
inline
void
CellArrayNeighbors<_Float, _D, _Record, _Location>::
initialize(_InputIterator begin, _InputIterator end)
{
  // Clear results from previous queries.
  _recordData.clear();
  // Dispense with the trivial case.
  if (begin == end) {
    return;
  }
  // Copy the records and compute the locations.
  RecLoc rl;
  while (begin != end) {
    rl.record = begin++;
    rl.location = _location(rl.record);
    _recordData.push_back(rl);
  }
  // Bound the locations.
  BBox box = {_recordData[0].location, _recordData[0].location};
  for (std::size_t i = 1; i != _recordData.size(); ++i) {
    box += _recordData[i].location;
  }
  // Expand to avoid errors in converting locations to cell multi-indices.
  offset(&box, std::sqrt(std::numeric_limits<Float>::epsilon()) *
         (Float(1) + ext::max(box.upper - box.lower)));
  // Set the lower corner and the cell array extents.
  _lowerCorner = box.lower;
  _cellArray.rebuild(computeExtentsAndSizes(_recordData.size(), box));
  // Sort by the cell indices and define the cells.
  cellSort();
}


template<typename _Float, std::size_t _D, typename _Record, typename _Location>
template<typename _OutputIterator>
inline
void
CellArrayNeighbors<_Float, _D, _Record, _Location>::
neighborQuery(const Point& center, const Float radius,
              _OutputIterator neighbors) const
{
  // Check trivial case.
  if (_recordData.empty()) {
    return;
  }
#ifdef STLIB_DEBUG
  // Check that the records were initialized.
  assert(_lowerCorner[0] == _lowerCorner[0]);
#endif
  typedef container::SimpleMultiIndexRange < D - 1 > Range;
  typedef container::SimpleMultiIndexRangeIterator < D - 1 > RangeIterator;

  // The window for the ORQ has corners at center - radius and
  // center + radius. Convert the corners to cell array indices.
  IndexList lo = locationToIndices(center - radius);
  IndexList hi = locationToIndices(center + radius);
  // Iterate over cells in all except the first dimension.
  typename Range::IndexList extents, bases;
  for (std::size_t i = 0; i != D - 1; ++i) {
    extents[i] = hi[i + 1] - lo[i + 1] + 1;
    bases[i] = lo[i + 1];
  }
  IndexList start, stop;
  start[0] = lo[0];
  stop[0] = hi[0] + 1;
  const Float squaredRadius = radius * radius;
  const Range range = {extents, bases};
  const RangeIterator end = RangeIterator::end(range);
  for (RangeIterator i = RangeIterator::begin(range); i != end; ++i) {
    for (std::size_t d = 0; d != D - 1; ++d) {
      start[d + 1] = stop[d + 1] = (*i)[d];
    }
    // Iterate over the records in the row.
    const ConstIterator jEnd = _cellArray(stop);
    for (ConstIterator j = _cellArray(start); j != jEnd; ++j) {
      if (ext::squaredDistance(center, j->location) < squaredRadius) {
        *neighbors++ = j->record;
      }
    }
  }
}


template<typename _Float, std::size_t _D, typename _Record, typename _Location>
inline
typename CellArrayNeighbors<_Float, _D, _Record, _Location>::IndexList
CellArrayNeighbors<_Float, _D, _Record, _Location>::
locationToIndices(const Point& x) const
{
  IndexList index;
  for (std::size_t i = 0; i != D; ++i) {
    index[i] = std::min(_cellArray.extents()[i] - 1,
                        Index(std::max(_Float(0), x[i] - _lowerCorner[i]) *
                              _inverseCellLengths[i]));
  }
  // Adjust to map to a non-empty cell.
  if (index[0] == _cellArray.extents()[0] - 1) {
    --index[0];
  }
  return index;
}

template<typename _Float, std::size_t _D, typename _Record, typename _Location>
inline
typename CellArrayNeighbors<_Float, _D, _Record, _Location>::Index
CellArrayNeighbors<_Float, _D, _Record, _Location>::
containerIndex(const Point& x)
{
  IndexList index;
  for (std::size_t i = 0; i != D; ++i) {
    index[i] = Index((x[i] - _lowerCorner[i]) * _inverseCellLengths[i]);
  }
  return _cellArray.arrayIndex(index);
}


template<typename _Float, std::size_t _D, typename _Record, typename _Location>
inline
void
CellArrayNeighbors<_Float, _D, _Record, _Location>::
cellSort()
{
  // Calculate the cell container index for each record.
  _cellIndices.resize(_recordData.size());
  // Count the number of records in each cell.
  _cellCounts.resize(_cellArray.size());
  // Initialize to zero.
  memset(&_cellCounts[0], 0, _cellCounts.size() * sizeof(std::size_t));
  for (std::size_t i = 0; i != _recordData.size(); ++i) {
    _cellIndices[i] = containerIndex(_recordData[i].location);
    ++_cellCounts[_cellIndices[i]];
  }
  // Turn the counts into end delimiters.
  std::partial_sum(_cellCounts.begin(), _cellCounts.end(),
                   _cellCounts.begin());
  // Copy the record data.
  _recordDataCopy.swap(_recordData);
  _recordData.resize(_recordDataCopy.size());
  // Define the cells.
  _cellArray[0] = _recordData.begin();
  for (std::size_t i = 1; i != _cellArray.size(); ++i) {
    _cellArray[i] = _recordData.begin() + _cellCounts[i - 1];
  }
  // Sort by the cell indices.
  for (std::size_t i = 0; i != _recordData.size(); ++i) {
    _recordData[--_cellCounts[_cellIndices[i]]] = _recordDataCopy[i];
  }
}

#if 0
// No longer used.
template<typename _Float, std::size_t _D, typename _Record, typename _Location>
inline
void
CellArrayNeighbors<_Float, _D, _Record, _Location>::
plainSort()
{
  // Calculate the cell container index for each record.
  for (std::size_t i = 0; i != _recordData.size(); ++i) {
    _recordData[i].cellIndex = containerIndex(_recordData[i].location);
  }
  // Sort by the cell indices.
  CompareIndex compare;
  std::sort(_recordData.begin(), _recordData.end(), compare);
  // Define the cells.
  {
    ConstIterator iter = _recordData.begin();
    for (std::size_t i = 0; i != _cellArray.size(); ++i) {
      _cellArray[i] = iter;
      while (iter != _recordData.end() && iter->cellIndex <= i) {
        ++iter;
      }
    }
  }
}
#endif


// Compute the array extents and the sizes for the cells.
template<typename _Float, std::size_t _D, typename _Record, typename _Location>
inline
typename CellArrayNeighbors<_Float, _D, _Record, _Location>::IndexList
CellArrayNeighbors<_Float, _D, _Record, _Location>::
computeExtentsAndSizes(const std::size_t numberOfCells, const BBox& domain)
{
  assert(numberOfCells > 0);

  // Work from the the least to greatest Cartesian extent to compute the
  // grid extents.
  IndexList extents;
  IndexList order;
  Point ext = domain.upper - domain.lower;
  ads::computeOrder(ext.begin(), ext.end(), order.begin());
  for (std::size_t i = 0; i != D; ++i) {
    // Normalize the domain to numberOfCells content.
    Float content = ext::product(ext);
    assert(content != 0);
    const Float factor = std::pow(numberOfCells / content,
                                  Float(1.0 / (D - i)));
    for (std::size_t j = i; j != D; ++j) {
      ext[order[j]] *= factor;
    }
    // The current index;
    const std::size_t n = order[i];
    // Add 0.5 and truncate to round to the nearest integer.
    ext[n] = extents[n] = std::max(std::size_t(1),
                                   std::size_t(ext[n] + 0.5));
  }

  // From the domain and the cell array extents, compute the cell size.
  for (std::size_t n = 0; n != D; ++n) {
    const Float d = domain.upper[n] - domain.lower[n];
    if (d == 0 || extents[n] == 1) {
      // The cell covers the entire domain.
      _inverseCellLengths[n] = 0;
    }
    else {
      _inverseCellLengths[n] = extents[n] / d;
    }
  }

  // Add one cell in the first dimension so that we will have convenient
  // end iterators when doing the window queries.
  ++extents[0];

  return extents;
}


} // namespace geom
}
