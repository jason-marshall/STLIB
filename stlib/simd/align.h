// -*- C++ -*-

#ifndef stlib_simd_align_h
#define stlib_simd_align_h

#include "stlib/simd/macros.h"

#include <boost/config.hpp>

#include <cstddef>

// C++11 has the alignas specifier.
#if (__cplusplus >= 201103L)

#define ALIGNAS(x) alignas(x)

#else

// Without alignas, we need a compiler-dependent solution.
#if defined(_MSC_VER)
#define ALIGNAS(x) __declspec(align(x))
#else // __GNUC__
#define ALIGNAS(x) __attribute__((aligned(x)))
#endif

#endif

namespace stlib
{
namespace simd
{


#ifdef STLIB_AVX512F
/// The alignment for SIMD operations with the default vector type.
BOOST_STATIC_CONSTEXPR std::size_t Alignment = 64;
/// Macro for aligning variables and structures.
#define ALIGN_SIMD ALIGNAS(64)
#elif defined(__AVX__)
BOOST_STATIC_CONSTEXPR std::size_t Alignment = 32;
#define ALIGN_SIMD ALIGNAS(32)
#elif defined(__SSE__)
BOOST_STATIC_CONSTEXPR std::size_t Alignment = 16;
#define ALIGN_SIMD ALIGNAS(16)
#else
BOOST_STATIC_CONSTEXPR std::size_t Alignment = 1;
#define ALIGN_SIMD
#endif


/// Return true if the pointer is aligned according to the native SIMD vector width.
inline
bool
isAligned(void const* p)
{
  return std::size_t(p) % Alignment == 0;
}


} // namespace simd
} // namespace stlib


#endif // stlib_simd_align_h
