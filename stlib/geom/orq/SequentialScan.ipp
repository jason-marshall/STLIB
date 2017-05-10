// -*- C++ -*-

#if !defined(__geom_SequentialScan_ipp__)
#error This file is an implementation detail of the class SequentialScan.
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
SequentialScan<N, _Location>::
computeWindowQuery(_OutputIterator output,
                   const typename Base::BBox& window) const
{
  std::size_t count = 0;
  ConstIterator iter = _records.begin();
  const ConstIterator iterEnd = _records.end();
  for (; iter != iterEnd; ++iter) {
    if (isInside(window, Base::_location(*iter))) {
      *(output++) = *iter;
      ++count;
    }
  }
  return count;
}


//
// File IO
//


template<std::size_t N, typename _Location>
inline
void
SequentialScan<N, _Location>::
put(std::ostream& out) const
{
  for (ConstIterator i = _records.begin(); i != _records.end(); ++i) {
    out << Base::_location(*i) << '\n';
  }
}

} // namespace geom
}
