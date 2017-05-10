// -*- C++ -*-

#if !defined(__geom_CellSearch_ipp__)
#error This file is an implementation detail of the class CellSearch.
#endif

namespace stlib
{
namespace geom
{

//
// Constructors
//


template<std::size_t N, typename _Location, typename _Cell>
inline
CellSearch<N, _Location, _Cell>::
CellSearch(const typename Base::Point& delta,
           const typename Base::BBox& domain) :
  Base(),
  _domain(),
  _cellArray(),
  _inverseCellSizes()
{
  setDomain(domain);
  computeExtentsAndSizes(delta);
}


template<std::size_t N, typename _Location, typename _Cell>
inline
CellSearch<N, _Location, _Cell>::
CellSearch(const typename Base::Point& delta, const typename Base::BBox& domain,
           typename Base::Record first, typename Base::Record last) :
  Base(),
  _domain(),
  _cellArray(),
  _inverseCellSizes()
{
  setDomain(domain);
  computeExtentsAndSizes(delta);
  insert(first, last);
  sort();
  initialize();
}


template<std::size_t N, typename _Location, typename _Cell>
inline
CellSearch<N, _Location, _Cell>::
CellSearch(const typename Base::Point& delta, typename Base::Record first,
           typename Base::Record last) :
  Base(),
  _domain(),
  _cellArray(),
  _inverseCellSizes()
{
  // Determine a domain to contain the records.
  computeDomain(first, last);
  computeExtentsAndSizes(delta);
  insert(first, last);
  sort();
  initialize();
}


//
// Memory usage.
//


template<std::size_t N, typename _Location, typename _Cell>
inline
std::size_t
CellSearch<N, _Location, _Cell>::
getMemoryUsage() const
{
  std::size_t usage = 0;
  // The (N-1)-D array of search structures.
  usage += sizeof(CellSearch);
  // For each search structure.
  for (typename DenseArray::const_iterator cellIterator = _cellArray.begin();
       cellIterator != _cellArray.end(); ++cellIterator) {
    usage += cellIterator->getMemoryUsage();
  }
  return usage;
}


//
// Window queries.
//


template<std::size_t N, typename _Location, typename _Cell>
inline
void
CellSearch<N, _Location, _Cell>::
sort()
{
  // Loop over the search structures.
  for (typename DenseArray::iterator i = _cellArray.begin();
       i != _cellArray.end(); ++i) {
    // Sort the records in the final coordinate.
    i->sort();
  }
}


template<std::size_t N, typename _Location, typename _Cell>
inline
void
CellSearch<N, _Location, _Cell>::
initialize()
{
  // Loop over the search structures.
  for (typename DenseArray::iterator i = _cellArray.begin();
       i != _cellArray.end(); ++i) {
    i->initialize();
  }
}


//
// File I/O
//


template<std::size_t N, typename _Location, typename _Cell>
inline
void
CellSearch<N, _Location, _Cell>::
put(std::ostream& out) const
{
  out.precision(16);
  out << "cell array domain = " << _domain
      << '\n'
      << "cell array size = " << _cellArray.extents()
      << '\n'
      << "inverse cell sizes = " << _inverseCellSizes
      << '\n';

  typename Cell::const_iterator recordIterator;

  // Loop over the search structures.
  for (typename DenseArray::const_iterator cellIterator = _cellArray.begin();
       cellIterator != _cellArray.end(); ++cellIterator) {
    for (recordIterator = cellIterator->begin();
         recordIterator != cellIterator->end(); ++recordIterator) {
      out << Base::_location(*recordIterator) << '\n';
    }
  }
}


//
// Private Member Functions
//


template<std::size_t N, typename _Location, typename _Cell>
inline
void
CellSearch<N, _Location, _Cell>::
computeExtentsAndSizes(const typename Base::Point& suggestedCellSize)
{
  typename DenseArray::SizeList extents;
  for (std::size_t n = 0; n != N - 1; ++n) {
    assert(suggestedCellSize[n] > 0);
    typename Base::Float cartesianExtent =
      _domain.upper[n] - _domain.lower[n];
    // Compute the cell array extents. Add 1 to round up.
    extents[n] = std::size_t(cartesianExtent / suggestedCellSize[n]) + 1;
    // From the domain and the cell array extents, compute the cell size.
    if (cartesianExtent == 0 || extents[n] == 1) {
      // The cell covers the entire domain.
      _inverseCellSizes[n] = 0;
    }
    else {
      _inverseCellSizes[n] = extents[n] / cartesianExtent;
    }
  }
  _cellArray.rebuild(extents);
}

template<std::size_t N, typename _Location, typename _Cell>
inline
void
CellSearch<N, _Location, _Cell>::
setDomain(const typename Base::BBox& domain)
{
  for (std::size_t n = 0; n != N - 1; ++n) {
    _domain.lower[n] = domain.lower[n];
    _domain.upper[n] = domain.upper[n];
  }
}

} // namespace geom
}
