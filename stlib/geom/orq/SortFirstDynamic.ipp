// -*- C++ -*-

#if !defined(__geom_SortFirstDynamic_ipp__)
#error This file is an implementation detail of the class SortFirstDynamic.
#endif

namespace stlib
{
namespace geom
{


//
// Window queries.
//


template<std::size_t N, typename _Location>
template<typename _OutputIterator>
inline
std::size_t
SortFirstDynamic<N, _Location>::
computeWindowQuery(_OutputIterator output,
                   const typename Base::BBox& window) const
{
  ConstIterator i = _records.lower_bound(window.lower[0]);
  const ConstIterator iEnd = _records.end();

  const typename Base::Float upperBound = window.upper[0];
  std::size_t count = 0;
  for (; i != iEnd && i->first <= upperBound; ++i) {
    if (isInside(window, Base::_location(i->second))) {
      *output++ = i->second;
      ++count;
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
SortFirstDynamic<N, _Location>::
put(std::ostream& out) const
{
  for (ConstIterator i = _records.begin(); i != _records.end(); ++i) {
    out << Base::_location(i->second) << '\n';
  }
}

} // namespace geom
}
