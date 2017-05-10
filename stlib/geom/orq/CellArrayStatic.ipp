// -*- C++ -*-

#if !defined(__geom_CellArrayStatic_ipp__)
#error This file is an implementation detail of the class CellArrayStatic.
#endif

namespace stlib
{
namespace geom
{

//
// Mathematical member functions
//

template<std::size_t N, typename _Location>
template<class _OutputIterator>
inline
std::size_t
CellArrayStatic<N, _Location>::
computeWindowQuery(_OutputIterator iter,
                   const typename Base::BBox& window) const
{
#if 0
  // CONTINUE
  std::cerr << "computeWindowQuery\n"
            << "window = " << window << '\n';
#endif
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

  // The interior portion of the index window for the N-1 last coordinates.
  typename Base::Point lower, upper;
  lower[0] = lo[0];
  upper[0] = hi[0];
  for (std::size_t n = 1; n != N; ++n) {
    lower[n] = lo[n] + 1;
    upper[n] = hi[n] - 1;
  }
  typename Base::BBox interior = {lower, upper};
#if 0
  // CONTINUE
  std::cerr << "lo = " << lo << '\n'
            << "hi = " << hi << '\n'
            << "interior = " << interior << '\n';
#endif

  // The number of records in the window.
  std::size_t count = 0;
  typename container::StaticArrayOfArrays<typename Base::Record>::const_iterator
  recordIter, recordIterEnd;

  //
  // Iterate over the cells in the index window.
  //

  typename Base::Point p;
  std::size_t n = N - 1;
  mi = lo;
  while (mi[N - 1] <= hi[N - 1]) {
    if (n == 0) {
      const std::size_t index = ext::dot(mi, _strides);
#if 0
      // CONTINUE
      std::cerr << "mi = " << mi << '\n'
                << "index = " << index << '\n';
#endif
      recordIter = _cellArray.begin(index);
      recordIterEnd = _cellArray.end(index + hi[0] - lo[0]);
      // Convert the index to a Cartesian point.
      for (std::size_t i = 0; i != p.size(); ++i) {
        p[i] = mi[i];
      }
      // If this is an interior cell.
      if (isInside(interior, p)) {
        for (; recordIter != recordIterEnd; ++recordIter) {
          // We only need to check the first coordinate.
          if (window.lower[0] <= Base::_location(*recordIter)[0]
              && Base::_location(*recordIter)[0] <=
              window.upper[0]) {
            *iter = *recordIter;
            ++iter;
            ++count;
          }
        }
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
      // We are done with this pass through the first coordinate.
      mi[0] = hi[0] + 1;
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
CellArrayStatic<N, _Location>::
put(std::ostream& out) const
{
  Base::put(out);
  // Write the multi-keys.
  for (typename
       container::StaticArrayOfArrays<typename Base::Record>::const_iterator
       i = _cellArray.begin(); i != _cellArray.end(); ++i) {
    out << Base::_location(*i) << '\n';
  }
}

} // namespace geom
}
