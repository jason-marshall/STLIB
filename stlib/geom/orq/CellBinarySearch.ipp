// -*- C++ -*-

#if !defined(__geom_CellBinarySearch_ipp__)
#error This file is an implementation detail of the class CellBinarySearch.
#endif

namespace stlib
{
namespace geom
{

//
// Mathematical member functions
//

template<std::size_t N, typename _Location>
template<typename _OutputIterator>
inline
std::size_t
CellBinarySearch<N, _Location>::
computeWindowQuery(_OutputIterator iter,
                   const typename Base::BBox& window) const
{
  //
  // Convert the multi-key to array indices.
  //
  typename Base::IndexList mi, lo, hi;
  Base::convertMultiKeyToIndices(window.lower, &lo);
  Base::convertMultiKeyToIndices(window.upper, &hi);


  //
  // Truncate the index window to lie within the cell array.
  //
  for (std::size_t n = 0; n != N - 1; ++n) {
    lo[n] = std::max(typename Base::Index(0), lo[n]);
    hi[n] = std::min(typename Base::Index(Base::getCellArrayExtents()[n] - 1),
                     hi[n]);
  }

  // The interior portion of the index window.
  geom::BBox < typename Base::Index, N - 1 > interior =
  {lo + typename Base::Index(1), hi - typename Base::Index(1)};

  // The number of records in the window.
  std::size_t count = 0;
  typename Base::Cell::const_iterator recordIterator;
  typename Base::Cell::const_iterator recordIteratorEnd;

  //
  // Iterate over the cells in the index window.
  //

  std::array < typename Base::Index, N - 1 > indexPoint;
  const typename Base::Float coordMin = window.lower[N - 1];
  const typename Base::Float coordMax = window.upper[N - 1];
  std::size_t n = N - 2;
  mi = lo;
  while (mi[N - 2] <= hi[N - 2]) {
    if (n == 0) {
      for (mi[0] = lo[0]; mi[0] <= hi[0]; ++mi[0]) {
        // Iterate over the records in the cell.
        const typename Base::Cell& cell = Base::getCell(mi);
        recordIteratorEnd = cell.end();

        // Binary search to find the beginning of the records in window.
        recordIterator = cell.search(coordMin);

        // Project to a lower dimension.
        for (std::size_t i = 0; i != indexPoint.size(); ++i) {
          indexPoint[i] = mi[i];
        }
        // If this is an interior cell.
        if (isInside(interior, indexPoint)) {
          for (; recordIterator != recordIteratorEnd &&
               Base::_location(*recordIterator)[N - 1] <= coordMax;
               ++recordIterator) {
            // There is no need to check if the record is in the window.
            *iter = *recordIterator;
            ++iter;
            ++count;
          }
        }
        else { // This is a boundary cell.
          for (; recordIterator != recordIteratorEnd &&
               Base::_location(*recordIterator)[N - 1] <= coordMax;
               ++recordIterator) {
            if (isInside(window, Base::_location(*recordIterator))) {
              *iter = *recordIterator;
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

} // namespace geom
}
