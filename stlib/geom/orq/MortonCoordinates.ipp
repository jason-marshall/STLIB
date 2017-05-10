// -*- C++ -*-

#if !defined(__geom_orq_MortonCoordinates_ipp__)
#error This file is an implementation detail of the class MortonCoordinates.
#endif

namespace stlib
{
namespace geom
{

//--------------------------------------------------------------------------
// Constructors etc.

template<typename _Float, std::size_t _Dimension>
inline
MortonCoordinates<_Float, _Dimension>::
MortonCoordinates(const BBox<Float, Dimension>& domain) :
  _lowerCorner(domain.lower),
  // The length of the box with equal sides that contains the domain.
  _length(ext::max(domain.upper - domain.lower)),
  _inverseLength(1. / _length)
{
  assert(_inverseLength > 0);
}

//--------------------------------------------------------------------------
// Functor.

template<typename _Float, std::size_t _Dimension>
inline
typename MortonCoordinates<_Float, _Dimension>::result_type
MortonCoordinates<_Float, _Dimension>::
operator()(argument_type p) const
{
  // Scale to the unit box.
  p -= _lowerCorner;
  p *= _inverseLength;
  // Truncate to lie within the unit box.
  for (std::size_t i = 0; i != p.size(); ++i) {
    if (p[i] < 0) {
      p[i] = 0;
    }
    if (p[i] >= 1) {
      p[i] = 1. - std::numeric_limits<_Float>::epsilon();
    }
  }
  // Convert to integer block coordinates.
  result_type coords;
  for (std::size_t i = 0; i != coords.size(); ++i) {
    coords[i] = std::size_t(p[i] * Extent);
  }
  return coords;
}

// Return the index of the highest level whose Morton box length exceeds the
// specified length.
template<typename _Float, std::size_t _Dimension>
inline
std::size_t
MortonCoordinates<_Float, _Dimension>::
level(const Float length) const
{
  Float boxLength = _length;
  std::size_t level = 0;
  for (; level < Levels; ++level) {
    boxLength *= 0.5;
    if (boxLength < length) {
      break;
    }
  }
  return level;
}

} // namespace geom
}
