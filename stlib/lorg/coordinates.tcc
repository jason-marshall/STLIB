// -*- C++ -*-

#if !defined(__lorg_coordinates_tcc__)
#error This file is an implementation detail of coordinates.
#endif

namespace stlib
{
namespace lorg
{

//--------------------------------------------------------------------------
// Constructors etc.

template<typename _Integer, typename _Float, std::size_t _Dimension>
inline
DiscreteCoordinates<_Integer, _Float, _Dimension>::
DiscreteCoordinates(const std::vector<Point>& positions) :
  _lowerCorner(),
  _scaling()
{
  // There must be at least one point.
  assert(! positions.empty());
  // Calculate a bounding box for the positions.
  geom::BBox<_Float, _Dimension> domain =
    geom::specificBBox<geom::BBox<_Float, _Dimension> >(positions.begin(),
                                                        positions.end());
  _lowerCorner = domain.lower;

  // The maximum length. Expand so that points will not lie on the upper sides
  // of the box.
  _Float maxLength = ext::max(domain.upper - domain.lower) *
                     (1 + 10 * std::numeric_limits<_Float>::epsilon());
  // Check for problems with the domain.
  if (maxLength == 0) {
    maxLength = 1;
  }

  // The scaling to convert offsets to discrete coordinates.
  // 2^numLevels / maxLength.
  _scaling = std::pow(2., double(NumLevels)) / maxLength;
}


//--------------------------------------------------------------------------
// Functor.

template<typename _Integer, typename _Float, std::size_t _Dimension>
inline
typename DiscreteCoordinates<_Integer, _Float, _Dimension>::DiscretePoint
DiscreteCoordinates<_Integer, _Float, _Dimension>::
discretize(const Point& p) const
{
  // Convert to integer block coordinates.
  DiscretePoint coords;
  for (std::size_t i = 0; i != coords.size(); ++i) {
    // Scale to the unit box and then to the array of cells. Then
    // cast to an integer. Note that we do not check if the points lies
    // in the semi-open bounding box.
    coords[i] = _Integer((p[i] - _lowerCorner[i]) * _scaling);
  }
  return coords;
}

} // namespace lorg
}
