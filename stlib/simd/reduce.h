// -*- C++ -*-

#ifndef stlib_simd_reduce_h
#define stlib_simd_reduce_h

#include "stlib/simd/align.h"
#include "stlib/simd/functions.h"

#include <cassert>

namespace stlib
{
namespace simd
{


/// Return the minimum value in the range.
/** The range must be aligned and padded. That is, both the beginning and end
    of the range must lie on SIMD vector boundaries. If the range is empty, 
    the result is \c std::numeric_limits<_Float>::infinity(). */
template<typename _Float>
_Float
minAlignedPadded(_Float const* begin, _Float const* end);


/// Return the minimum value in the range.
/** The range must be aligned, padded, and non-empty. That is, both the
    beginning and end of the range must lie on SIMD vector boundaries. */
template<typename _Float>
_Float
minAlignedPaddedNonEmpty(_Float const* begin, _Float const* end);


} // namespace simd
} // namespace stlib

#define stlib_simd_reduce_tcc
#include "stlib/simd/reduce.tcc"
#undef stlib_simd_reduce_tcc


#endif // stlib_simd_reduce_h
