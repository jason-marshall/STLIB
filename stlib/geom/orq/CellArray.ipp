// -*- C++ -*-

#if !defined(__geom_CellArray_ipp__)
#error This file is an implementation detail of the class CellArray.
#endif

namespace stlib
{
namespace geom
{


//
// Memory usage.
//


template<std::size_t N, typename _Location>
inline
std::size_t
CellArray<N, _Location>::
getMemoryUsage() const
{
  std::size_t usage = 0;
  for (typename DenseArray::const_iterator i = _cellArray.begin();
       i != _cellArray.end(); ++i) {
    usage += i->size() * sizeof(typename Base::Record);
  }
  usage += _cellArray.size() * sizeof(typename Base::Cell);
  return usage;
}


//
// Mathematical member functions
//


template<std::size_t N, typename _Location>
template<class _OutputIterator>
inline
std::size_t
CellArray<N, _Location>::
computeWindowQuery(_OutputIterator iter,
                   const typename Base::BBox& window) const
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

  {
    for (std::size_t n = 0; n != N; ++n) {
      lo[n] = std::max(typename Base::Index(0), lo[n]);
      hi[n] = std::min(typename Base::Index(Base::getExtents()[n] - 1),
                       hi[n]);
    }
  }

  // The interior portion of the index window.
  typename Base::Point lower;
  typename Base::Point upper;
  for (std::size_t n = 0; n != N; ++n) {
    lower[n] = lo[n] + 1;
    upper[n] = hi[n] - 1;
  }
  typename Base::BBox interior = {lower, upper};

  // The number of records in the window.
  std::size_t count = 0;
  typename Base::Cell::const_iterator recordIter;
  typename Base::Cell::const_iterator recordIterEnd;

  //
  // Iterate over the cells in the index window.
  //

  typename Base::Point p;
  std::size_t n = N - 1;
  mi = lo;
  while (mi[N - 1] <= hi[N - 1]) {
    if (n == 0) {
      for (mi[0] = lo[0]; mi[0] <= hi[0]; ++mi[0]) {
        // Iterate over the records in the cell.
        const typename Base::Cell& cell = _cellArray(mi);
        recordIter = cell.begin();
        recordIterEnd = cell.end();

        // Convert the index to a Cartesian point.
        for (std::size_t i = 0; i != p.size(); ++i) {
          p[i] = mi[i];
        }
        // If this is an interior cell.
        if (isInside(interior, p)) {
          for (; recordIter != recordIterEnd; ++recordIter) {
            // No need to check if it is in the window.
            *iter = *recordIter;
            ++iter;
          }
          count += std::size_t(cell.size());
        }
        else {
          for (; recordIter != recordIterEnd; ++recordIter) {
            // If the record is in the window.
            if (isInside(window, Base::_location(*recordIter))) {
              *iter = *recordIter;
              ++iter;
              ++count;
            }
          }
        }
      }
      ++n;
    }
    else if (mi[n - 1] > hi[n - 1]) {
      mi[n - 1] = lo[n - 1];
      ++mi[n];
      ++n;
    }
    else {
      --n;
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
CellArray<N, _Location>::
put(std::ostream& out) const
{
  Base::put(out);

  for (typename DenseArray::const_iterator i = _cellArray.begin();
       i != _cellArray.end(); ++i) {
    const typename Base::Cell& b = *i;
    typename Base::Cell::const_iterator iter(b.begin());
    while (iter != b.end()) {
      out << Base::_location(*(iter++)) << '\n';
    }
  }
}

} // namespace geom
}
