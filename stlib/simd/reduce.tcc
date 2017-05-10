// -*- C++ -*-

#if !defined(stlib_simd_reduce_tcc)
#error This file is an implementation detail of reduce.
#endif

namespace stlib
{
namespace simd
{


template<typename _Float>
inline
_Float
minAlignedPadded(_Float const* begin, _Float const* end)
{
#ifdef STLIB_DEBUG
  assert(isAligned(begin));
  assert(isAligned(end));
#endif
  typename Vector<_Float>::Type result =
    set1(std::numeric_limits<_Float>::infinity());
  for ( ; begin != end; begin += Vector<_Float>::Size) {
    result = min(result, load(begin));
  }
  return min(result);
}


template<typename _Float>
inline
_Float
minAlignedPaddedNonEmpty(_Float const* begin, _Float const* end)
{
#ifdef STLIB_DEBUG
  assert(isAligned(begin));
  assert(isAligned(end));
  assert(begin < end);
#endif
  typename Vector<_Float>::Type result = load(begin);
  for (begin += Vector<_Float>::Size; begin != end;
       begin += Vector<_Float>::Size) {
    result = min(result, load(begin));
  }
  return min(result);
}


} // namespace simd
} // namespace stlib
