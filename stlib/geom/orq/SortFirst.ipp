// -*- C++ -*-

#if !defined(__geom_SortFirst_ipp__)
#error This file is an implementation detail of the class SortFirst.
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
SortFirst<N, _Location>::
computeWindowQuery(_OutputIterator output,
                   const typename Base::BBox& window) const
{
  ConstIterator i = std::lower_bound(_sorted.begin(), _sorted.end(),
                                     window.lower,
                                     _lessThanCompareValueAndMultiKey);
  const ConstIterator iEnd =
    std::upper_bound(_sorted.begin(), _sorted.end(), window.upper,
                     _lessThanCompareMultiKeyAndValue);

  std::size_t count = 0;
  for (; i != iEnd; ++i) {
    if (isInside(window, Base::_location(*i))) {
      *output++ = *i;
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
SortFirst<N, _Location>::
put(std::ostream& out) const
{
  for (ConstIterator i = _sorted.begin(); i != _sorted.end(); ++i) {
    out << Base::_location(*i) << '\n';
  }
}


//
// Validity check.
//

template<std::size_t N, typename _Location>
inline
bool
SortFirst<N, _Location>::
isValid() const
{
  if (! std::is_sorted(_sorted.begin(), _sorted.end(), _lessThanCompare)) {
    return false;
  }
  return true;
}

} // namespace geom
}
