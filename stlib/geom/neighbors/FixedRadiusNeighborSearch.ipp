// -*- C++ -*-

#if !defined(__geom_FixedRadiusNeighborSearch_ipp__)
#error This file is an implementation detail of the class FixedRadiusNeighborSearch.
#endif

namespace stlib
{
namespace geom
{

template<std::size_t N, typename _Location>
template<class _IndexOutputIterator>
inline
void
FixedRadiusNeighborSearch<N, _Location>::
findNeighbors(_IndexOutputIterator iter, const std::size_t recordIndex)
{
  // Make a bounding box that contains the sphere.
  const typename Base::Point center =
    Base::_location(_recordsBegin + recordIndex);
  typename Base::Point p = center;
  p -= _radius;
  _boundingBox.lower = p;
  p = center;
  p += _radius;
  _boundingBox.upper = p;
  // Get the records that are in the box.
  _recordsInBox.clear();
  Base::computeWindowQuery(std::back_inserter(_recordsInBox), _boundingBox);
  // Select the records that are within the search radius (and are not the
  // specified record).
  std::size_t index;
  for (std::size_t i = 0; i != _recordsInBox.size(); ++i) {
    const typename Base::Record& r = _recordsInBox[i];
    index = std::distance(_recordsBegin, r);
    if (index != recordIndex &&
        ext::squaredDistance(center, Base::_location(r)) <= _squaredRadius) {
      *iter++ = index;
    }
  }
}

} // namespace geom
}
