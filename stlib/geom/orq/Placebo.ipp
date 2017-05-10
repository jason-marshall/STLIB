// -*- C++ -*-

#if !defined(__geom_Placebo_ipp__)
#error This file is an implementation detail of the class Placebo.
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
Placebo<N, _Location>::
computeWindowQuery(_OutputIterator iter,
                   const typename Base::BBox& /*window*/) const
{
  typename std::vector<typename Base::Record>::const_iterator recordIterator
    = _records.begin() + getStartingPoint();
  const typename std::vector<typename Base::Record>::const_iterator recordEnd
    = recordIterator + querySize;
  while (recordIterator != recordEnd) {
    *(iter++) = *(recordIterator++);
  }
  return querySize;
}


//
// File I/O
//


template<std::size_t N, typename _Location>
inline
void
Placebo<N, _Location>::
put(std::ostream& out) const
{
  for (typename std::vector<typename Base::Record>::const_iterator i =
         _records.begin(); i != _records.end(); ++i) {
    out << Base::_location(*i) << '\n';
  }
}

} // namespace geom
}
