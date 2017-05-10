// -*- C++ -*-

#if !defined(__geom_PlaceboCheck_ipp__)
#error This file is an implementation detail of the class PlaceboCheck.
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
PlaceboCheck<N, _Location>::
computeWindowQuery(_OutputIterator iter,
                   const typename Base::BBox& window) const
{
  std::size_t count = 0;
  typename std::vector<typename Base::Record>::const_iterator recordIterator
    = Base::_records.begin() + Base::getStartingPoint();
  const typename std::vector<typename Base::Record>::const_iterator recordEnd
    = recordIterator + Base::querySize;
  for (; recordIterator != recordEnd; ++recordIterator) {
    if (isInside(window, Base::_location(*recordIterator))) {
      *(iter++) = *recordIterator;
      ++count;
    }
  }
  return count;
}

} // namespace geom
} // namespace stlib
