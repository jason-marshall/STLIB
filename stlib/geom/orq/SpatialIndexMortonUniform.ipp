// -*- C++ -*-

#if !defined(__geom_orq_SpatialIndexMortonUniform_ipp__)
#error This file is an implementation detail of the class SpatialIndexMortonUniform.
#endif

namespace stlib
{
namespace geom
{

//--------------------------------------------------------------------------
// Constructors etc.

template<typename _Float, std::size_t _Dimension>
inline
SpatialIndexMortonUniform<_Float, _Dimension>::
SpatialIndexMortonUniform(const BBox<Float, Dimension>& domain,
                          const Float maximumLength) :
  _levels(0),
  _extent(1),
  _lowerCorner(domain.lower),
  _inverseLength()
{
  assert(maximumLength > 0);

  // The length of the box with equal sides that contains the domain.
  Float length = ext::max(domain.upper - domain.lower);
  _inverseLength = 1. / length;
  // Determine the number of levels of refinement.
  while (length > maximumLength) {
    ++_levels;
    _extent *= 2;
    length *= 0.5;
  }
  // Make sure that we have enough bits to store the interleaved code.
  assert(Dimension * _levels <=
         std::size_t(std::numeric_limits<std::size_t>::digits));
}

//--------------------------------------------------------------------------
// Functor.

template<typename _Float, std::size_t _Dimension>
inline
typename SpatialIndexMortonUniform<_Float, _Dimension>::result_type
SpatialIndexMortonUniform<_Float, _Dimension>::
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
      p[i] = 1. - std::numeric_limits<Float>::epsilon();
    }
  }
  // Scale to continuous block coordinates.
  p *= Float(_extent);
  // Convert to integer block coordinates.
  std::array<std::size_t, Dimension> coords;
  for (std::size_t i = 0; i != coords.size(); ++i) {
    coords[i] = std::size_t(p[i]);
  }
  // Interlace the coordinates to obtain the code.
  std::size_t code = 0;
  std::size_t mask = std::size_t(1) << (_levels - 1);
  for (std::size_t i = 0; i != _levels; ++i) {
    for (std::size_t j = 0; j != Dimension; ++j) {
      code <<= 1;
      code |= (mask & coords[Dimension - 1 - j]) >> (_levels - 1 - i);
    }
    mask >>= 1;
  }
  return code;
}

} // namespace geom
}
