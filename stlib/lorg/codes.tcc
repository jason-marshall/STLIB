// -*- C++ -*-

#if !defined(__lorg_codes_tcc__)
#error This file is an implementation detail of codes.
#endif

namespace stlib
{
namespace lorg
{


//----------------------------------------------------------------------------
// Morton.
//----------------------------------------------------------------------------

template<typename _Integer, typename _Float, std::size_t _Dimension>
inline
Morton<_Integer, _Float, _Dimension>::
Morton(const std::vector<typename Base::Point>& positions) :
  Base(positions),
  _expanded()
{
  buildExpanded();
}

template<typename _Integer, typename _Float, std::size_t _Dimension>
inline
void
Morton<_Integer, _Float, _Dimension>::
buildExpanded()
{
  // Make the array of expanded bits.
  _Integer mask;
  for (std::size_t i = 0; i != _expanded.size(); ++i) {
    _expanded[i] = 0;
    mask = 1;
    // Move each bit at position n to position _Dimension * n.
    for (std::size_t j = 0; j != Base::NumLevels; ++j) {
      _expanded[i] |= (mask & i) << (_Dimension - 1) * j;
      mask <<= 1;
    }
  }
}

template<typename _Integer, typename _Float, std::size_t _Dimension>
inline
_Integer
Morton<_Integer, _Float, _Dimension>::
expand(std::size_t n) const
{
  const _Integer mask = (1 << ExpandBits) - 1;
  _Integer result = 0;
  for (std::size_t i = 0; i < Base::NumLevels;
       i += ExpandBits, n >>= ExpandBits) {
    result |= _expanded[n & mask] << i * _Dimension;
  }
  return result;
}


} // namespace lorg
}
