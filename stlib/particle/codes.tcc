// -*- C++ -*-

#if !defined(__particle_codes_tcc__)
#error This file is an implementation detail of codes.
#endif

namespace stlib
{
namespace particle
{


//----------------------------------------------------------------------------
// Morton.
//----------------------------------------------------------------------------


template<typename _Float, std::size_t _Dimension, bool _Periodic>
inline
Morton<_Float, _Dimension, _Periodic>::
Morton(const geom::BBox<_Float, _Dimension>& domain, _Float cellLength) :
  Base(),
  _expanded(),
  _separated()
{
  initialize(domain, cellLength);
  // Make the array of separated bits.
  buildSeparated();
}


template<typename _Float, std::size_t _Dimension, bool _Periodic>
inline
Morton<_Float, _Dimension, _Periodic>::
Morton() :
  Base(),
  _expanded(),
  _separated()
{
  // Make the array of separated bits.
  buildSeparated();
}


template<typename _Float, std::size_t _Dimension, bool _Periodic>
inline
void
Morton<_Float, _Dimension, _Periodic>::
initialize(geom::BBox<_Float, _Dimension> domain, _Float cellLength)
{
  // The data structure for computing Morton coordinates.
  Base::initialize(domain, cellLength);
  // Check that there are enough bits in the Code data type.
  assert(Base::numLevels() <= (std::numeric_limits<Code>::digits - 1) /
         _Dimension);
  buildExpanded();
}


template<typename _Float, std::size_t _Dimension, bool _Periodic>
inline
void
Morton<_Float, _Dimension, _Periodic>::
setLevels(const std::size_t numLevels)
{
  Base::setLevels(numLevels);
  buildExpanded();
  buildSeparated();
}


template<typename _Float, std::size_t _Dimension, bool _Periodic>
inline
void
Morton<_Float, _Dimension, _Periodic>::
buildExpanded()
{
  // Make the array of expanded bits.
  Code mask;
  for (std::size_t i = 0; i != _expanded.size(); ++i) {
    _expanded[i] = 0;
    mask = 1;
    // Move each bit at position n to position _Dimension * n.
    for (std::size_t j = 0; j != ExpandBits; ++j) {
      _expanded[i] |= (mask & i) << (_Dimension - 1) * j;
      mask <<= 1;
    }
  }
}

template<typename _Float, std::size_t _Dimension, bool _Periodic>
inline
void
Morton<_Float, _Dimension, _Periodic>::
buildSeparated()
{
  // Separate the interleaved bits. 101 010 101 is transformed to
  // the coordinate (101, 010, 101).
  Code mask;
  for (std::size_t i = 0; i != _separated.size(); ++i) {
    // Start with the least significant bit.
    mask = 1;
    for (std::size_t k = 0; k != _Dimension; ++k) {
      _separated[i][k] = 0;
    }
    for (std::size_t j = 0; j != SeparateBits; ++j) {
      // For each coordinate.
      for (std::size_t k = 0; k != _Dimension; ++k) {
        _separated[i][k] |= (i & mask) >> (j * (_Dimension - 1) + k);
        mask <<= 1;
      }
    }
  }
}

template<typename _Float, std::size_t _Dimension, bool _Periodic>
inline
typename Morton<_Float, _Dimension, _Periodic>::DiscretePoint
Morton<_Float, _Dimension, _Periodic>::
coordinates(Code code) const
{
  const Code mask = (1 << _Dimension * SeparateBits) - 1;
  DiscretePoint coords = {{}};
  for (std::size_t i = 0; i * SeparateBits < Base::numLevels();
       ++i, code >>= _Dimension * SeparateBits) {
    for (std::size_t j = 0; j != _Dimension; ++j) {
      coords[j] |= typename Base::DiscreteCoordinate
                   (_separated[code & mask][j]) << i * SeparateBits;
    }
  }
  return coords;
}

template<typename _Float, std::size_t _Dimension, bool _Periodic>
inline
typename Morton<_Float, _Dimension, _Periodic>::Code
Morton<_Float, _Dimension, _Periodic>::
expand(Code n) const
{
  const Code mask = (1 << ExpandBits) - 1;
  Code result = 0;
  for (std::size_t i = 0; i < Base::numLevels();
       i += ExpandBits, n >>= ExpandBits) {
    result |= _expanded[n & mask] << i * _Dimension;
  }
  return result;
}


} // namespace particle
}
