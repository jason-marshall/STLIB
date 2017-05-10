// -*- C++ -*-

#if !defined(__sfc_MortonOrder_tcc__)
#error This file is an implementation detail of MortonOrder.
#endif

namespace stlib
{
namespace sfc
{


template<std::size_t _Dimension, typename _Code>
inline
_Code
MortonOrder<_Dimension, _Code>::
code(const std::array<_Code, _Dimension>& indices,
     const std::size_t numLevels) const
{
  _Code result = 0;
  for (std::size_t i = 0; i != _Dimension; ++i) {
    result |= _dilate(indices[i], numLevels) << i;
  }
  return result;
}


} // namespace sfc
} // namespace stlib
