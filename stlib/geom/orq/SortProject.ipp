// -*- C++ -*-

#if !defined(__geom_SortProject_ipp__)
#error This file is an implementation detail of the class SortProject.
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
SortProject<N, _Location>::
computeWindowQuery(_OutputIterator output,
                   const typename Base::BBox& window) const
{
  //
  // Get each slice.
  //

  std::array<ConstIterator, N> begin;
  std::array<ConstIterator, N> end;
  std::array<std::size_t, N> size;
  for (std::size_t n = 0; n != N; ++n) {
    _lessThanCompareValueAndMultiKey.set(n);
    _lessThanCompareMultiKeyAndValue.set(n);
    begin[n] = std::lower_bound(_sorted[n].begin(), _sorted[n].end(),
                                window.lower,
                                _lessThanCompareValueAndMultiKey);
    end[n] = std::upper_bound(_sorted[n].begin(), _sorted[n].end(),
                              window.upper,
                              _lessThanCompareMultiKeyAndValue);
    size[n] = end[n] - begin[n];
  }

  // Choose the smallest slice.
  std::size_t count = 0;
  const std::size_t n =
    std::min_element(size.begin(), size.end()) - size.begin();

  ConstIterator i;
  const ConstIterator iEnd = end[n];
  for (i = begin[n]; i != iEnd; ++i) {
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
SortProject<N, _Location>::
put(std::ostream& out) const
{
  for (ConstIterator i = _sorted[0].begin(); i != _sorted[0].end(); ++i) {
    out << Base::_location(*i) << '\n';
  }
}


//
// Validity check.
//

template<std::size_t N, typename _Location>
inline
bool
SortProject<N, _Location>::
isValid() const
{
  for (std::size_t n = 0; n != N; ++n) {
    _lessThanCompare.set(n);
    if (! std::is_sorted(_sorted[n].begin(), _sorted[n].end(),
                         _lessThanCompare)) {
      return false;
    }
  }
  return true;
}

} // namespace geom
} // namespace stlib
