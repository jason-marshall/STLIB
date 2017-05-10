// -*- C++ -*-

#if !defined(__sfc_DilateBits_tcc__)
#error This file is an implementation detail of DilateBits.
#endif

namespace stlib
{
namespace sfc
{

template<std::size_t _Dimension>
inline
DilateBits<_Dimension>::
DilateBits() :
  _expanded()
{
  // Check that there are enough bits in the _Code data type.
  static_assert(_Dimension * _ExpandBits <=
                std::size_t(std::numeric_limits<_DilatedInteger>::digits),
                "Insufficient digits.");
  // Make the array of expanded bits.
  _DilatedInteger mask;
  for (std::size_t i = 0; i != _expanded.size(); ++i) {
    _expanded[i] = 0;
    mask = 1;
    // Move each bit at position n to position _Dimension * n.
    for (std::size_t j = 0; j != _ExpandBits; ++j) {
      _expanded[i] |= (mask & i) << (_Dimension - 1) * j;
      mask <<= 1;
    }
  }
}

template<std::size_t _Dimension>
template<typename _Code>
inline
_Code
DilateBits<_Dimension>::
operator()(_Code n, const int nBits) const
{
  const _Code mask = (1 << _ExpandBits) - 1;
  _Code result = 0;
  for (int i = 0; i < nBits; i += _ExpandBits, n >>= _ExpandBits) {
    result |= _Code(_expanded[n & mask]) << i * _Dimension;
  }
  return result;
}

} // namespace sfc
}
