// -*- C++ -*-

#if !defined(__geom_SparseCellArray_ipp__)
#error This file is an implementation detail of the class SparseCellArray.
#endif

namespace stlib
{
namespace geom
{


//
// Constructors
//


template<std::size_t N, typename _Location>
inline
void
SparseCellArray<N, _Location>::
build()
{
  typename VectorArray::SizeList arrayExtents;
  for (std::size_t n = 0; n != N - 1; ++n) {
    arrayExtents[n] = Base::getExtents()[n];
  }
  _vectorArray.rebuild(arrayExtents);
}


//
// Memory usage.
//


template<std::size_t N, typename _Location>
inline
std::size_t
SparseCellArray<N, _Location>::
getMemoryUsage() const
{
  std::size_t usage = 0;
  // The (N-1)-D array of sparse vectors.
  usage += _vectorArray.size()
           * sizeof(SparseCellVector<Cell>);
  // For each sparse vector.
  for (typename VectorArray::const_iterator i = _vectorArray.begin();
       i != _vectorArray.end(); ++i) {
    // The memory for the structure of the cell.
    usage += i->size() *
             sizeof(IndexAndCell<Cell>);
    for (typename SparseCellVector<Cell>::
         const_iterator j = i->begin();
         j != i->end();
         ++j) {
      // The memory for the contents of the cell.
      usage += j->cell.size() * sizeof(typename Base::Record);
    }
  }
  return usage;
}


//
// Mathematical member functions
//


template<std::size_t N, typename _Location>
template<class _OutputIterator>
inline
std::size_t
SparseCellArray<N, _Location>::
computeWindowQuery(_OutputIterator iter,
                   const BBox& window) const
{
  //
  // Convert the multi-key to array index coordinates.
  //

  // A multi-index and the lower and upper bounds of the index window.
  typename Base::IndexList mi, lo, hi;
  Base::convertMultiKeyToIndices(window.lower, &lo);
  Base::convertMultiKeyToIndices(window.upper, &hi);

  //
  // Truncate the index window to lie within the cell array.
  //

  for (std::size_t n = 0; n != N; ++n) {
    lo[n] = std::max(typename Base::Index(0), lo[n]);
    hi[n] = std::min(typename Base::Index(Base::getExtents()[n] - 1), hi[n]);
  }

  // The interior portion of the index window.
  Point lower, upper;
  for (std::size_t n = 0; n != N; ++n) {
    lower[n] = lo[n] + 1;
    upper[n] = hi[n] - 1;
  }
  BBox interior = {lower, upper};

  // The number of records in the window.
  std::size_t count = 0;
  typename Cell::const_iterator recordIterator, recordIteratorEnd;
  typename SparseCellVector<Cell>::const_iterator cellIterator,
           cellIteratorEnd;

  //
  // Iterate over the cells in the index window.
  //

  Point p;
  // Array Index.
  std::array < typename Base::IndexList::value_type, N - 1 > ai;
  std::size_t n = N - 2;
  mi = lo;
  for (std::size_t i = 0; i != N - 1; ++i) {
    ai[i] = mi[i];
  }
  while (mi[N - 2] <= hi[N - 2]) {
    if (n == 0) {
      for (mi[0] = lo[0]; mi[0] <= hi[0]; ++mi[0]) {
        ai[0] = mi[0];
        const SparseCellVector<Cell>& cell_vector =
          _vectorArray(ai);
        cellIterator = cell_vector.lower_bound(lo[N - 1]);
        cellIteratorEnd = cell_vector.end();
        for (; cellIterator != cellIteratorEnd &&
             typename Base::Index(cellIterator->index) <= hi[N - 1];
             ++cellIterator) {
          const Cell& cell = cellIterator->cell;
          recordIterator = cell.begin();
          recordIteratorEnd = cell.end();

          mi[N - 1] = cellIterator->index;
          // Convert the index to a Cartesian point.
          for (std::size_t i = 0; i != p.size(); ++i) {
            p[i] = mi[i];
          }
          // If this is an interior cell.
          if (isInside(interior, p)) {
            for (; recordIterator != recordIteratorEnd; ++recordIterator) {
              // There is no need to check if the record is in the window.
              *iter = *recordIterator;
              ++iter;
            }
            count += cell.size();
          }
          else { // This is a boundary cell.
            for (; recordIterator != recordIteratorEnd; ++recordIterator) {
              if (isInside(window, Base::_location(*recordIterator))) {
                *iter = *recordIterator;
                ++iter;
                ++count;
              }
            }
          }
        }
      }
      ++n;
    }
    else if (mi[n - 1] > hi[n - 1]) {
      ai[n - 1] = mi[n - 1] = lo[n - 1];
      ++mi[n];
      ++ai[n];
      ++n;
    }
    else {
      --n;
    }
  }

  return count;
}


// Indexing by multi-key.  Return a reference to a cell.
template<std::size_t N, typename _Location>
inline
typename SparseCellArray<N, _Location>::Cell&
SparseCellArray<N, _Location>::
operator()(const Point& multiKey)
{
  // CONTINUE: this is not efficient.
  typename Base::IndexList mi;
  Base::convertMultiKeyToIndices(multiKey, &mi);
  std::array < typename Base::IndexList::value_type, N - 1 > ai;
  for (std::size_t n = 0; n != N - 1; ++n) {
    ai[n] = mi[n];;
  }
  return _vectorArray(ai).find(mi[N - 1]);
}


//
// File IO
//


template<std::size_t N, typename _Location>
inline
void
SparseCellArray<N, _Location>::
put(std::ostream& out) const
{
  Base::put(out);

  typename Cell::const_iterator recordIterator;
  typename SparseCellVector<Cell>::const_iterator
  vectorIterator;

  typename VectorArray::const_iterator
  i = _vectorArray.begin(),
  iEnd = _vectorArray.end();
  for (; i != iEnd; ++i) {
    const SparseCellVector<Cell>& cellVector = *i;
    vectorIterator = cellVector.begin();
    for (; vectorIterator != cellVector.end(); ++vectorIterator) {
      const Cell& b = vectorIterator->cell;
      recordIterator = b.begin();
      while (recordIterator != b.end()) {
        out << Base::_location(*(recordIterator++)) << '\n';
      }
    }
  }
}

} // namespace geom
}
