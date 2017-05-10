// -*- C++ -*-

#if !defined(stlib_simd_constants_h)
#define stlib_simd_constants_h

#include "stlib/simd/macros.h"

#include <boost/config.hpp>

#include <cstddef>

// When I support AVX-512F, change the macro STLIB_AVX512F to __AVX512F__.

#if defined(STLIB_AVX512F) || defined(__AVX__)
#include <immintrin.h>
#endif
#ifdef __SSE4_1__
#include <smmintrin.h>
#endif
#ifdef __SSE3__
#include <pmmintrin.h>
#endif
#ifdef __SSE2__
#include <emmintrin.h>
#endif
#ifdef __SSE__
#include <xmmintrin.h>
#endif

namespace stlib
{
namespace simd
{


/// The default size of SIMD short vectors.
template<typename _T>
struct DefaultVector;


#ifdef STLIB_AVX512F
template<>
struct DefaultVector<float> {
  BOOST_STATIC_CONSTEXPR std::size_t Size = 16;
};
template<>
struct DefaultVector<double> {
  BOOST_STATIC_CONSTEXPR std::size_t Size = 8;
};
template<>
struct DefaultVector<int> {
  BOOST_STATIC_CONSTEXPR std::size_t Size = 16;
};
template<>
struct DefaultVector<unsigned> {
  BOOST_STATIC_CONSTEXPR std::size_t Size = 16;
};
#elif defined(__AVX2__)
template<>
struct DefaultVector<float> {
  BOOST_STATIC_CONSTEXPR std::size_t Size = 8;
};
template<>
struct DefaultVector<double> {
  BOOST_STATIC_CONSTEXPR std::size_t Size = 4;
};
template<>
struct DefaultVector<int> {
  BOOST_STATIC_CONSTEXPR std::size_t Size = 8;
};
template<>
struct DefaultVector<unsigned> {
  BOOST_STATIC_CONSTEXPR std::size_t Size = 8;
};
#elif defined(__AVX__)
template<>
struct DefaultVector<float> {
  BOOST_STATIC_CONSTEXPR std::size_t Size = 8;
};
template<>
struct DefaultVector<double> {
  BOOST_STATIC_CONSTEXPR std::size_t Size = 4;
};
template<>
struct DefaultVector<int> {
  BOOST_STATIC_CONSTEXPR std::size_t Size = 4;
};
template<>
struct DefaultVector<unsigned> {
  BOOST_STATIC_CONSTEXPR std::size_t Size = 4;
};
#elif defined(__SSE2__)
template<>
struct DefaultVector<float> {
  BOOST_STATIC_CONSTEXPR std::size_t Size = 4;
};
template<>
struct DefaultVector<double> {
  BOOST_STATIC_CONSTEXPR std::size_t Size = 2;
};
template<>
struct DefaultVector<int> {
  BOOST_STATIC_CONSTEXPR std::size_t Size = 4;
};
template<>
struct DefaultVector<unsigned> {
  BOOST_STATIC_CONSTEXPR std::size_t Size = 4;
};
#elif defined(__SSE__)
template<>
struct DefaultVector<float> {
  BOOST_STATIC_CONSTEXPR std::size_t Size = 4;
};
template<>
struct DefaultVector<double> {
  BOOST_STATIC_CONSTEXPR std::size_t Size = 2;
};
template<>
struct DefaultVector<int> {
  BOOST_STATIC_CONSTEXPR std::size_t Size = 1;
};
template<>
struct DefaultVector<unsigned> {
  BOOST_STATIC_CONSTEXPR std::size_t Size = 1;
};
#else
template<>
struct DefaultVector<float> {
  BOOST_STATIC_CONSTEXPR std::size_t Size = 1;
};
template<>
struct DefaultVector<double> {
  BOOST_STATIC_CONSTEXPR std::size_t Size = 1;
};
template<>
struct DefaultVector<int> {
  BOOST_STATIC_CONSTEXPR std::size_t Size = 1;
};
template<>
struct DefaultVector<unsigned> {
  BOOST_STATIC_CONSTEXPR std::size_t Size = 1;
};
#endif


/// Constants related to a SIMD short vector.
template<typename _T, std::size_t _Size = DefaultVector<_T>::Size>
struct Vector;


/// Specialization for float.
template<>
struct Vector<float, 1> {
  /// The SIMD short vector type.
  typedef float Type;
  /// The length of a SIMD vector of single-precision, floating-point numbers.
  BOOST_STATIC_CONSTEXPR std::size_t Size = 1;
};
#ifdef STLIB_AVX512F
template<>
struct Vector<float, 16> {
  typedef __m512 Type;
  BOOST_STATIC_CONSTEXPR std::size_t Size = 16;
};
#endif
#ifdef __AVX__
template<>
struct Vector<float, 8> {
  typedef __m256 Type;
  BOOST_STATIC_CONSTEXPR std::size_t Size = 8;
};
#endif
#ifdef __SSE__
template<>
struct Vector<float, 4> {
  typedef __m128 Type;
  BOOST_STATIC_CONSTEXPR std::size_t Size = 4;
};
#endif


/// Specialization for double.
template<>
struct Vector<double, 1> {
  /// The SIMD short vector type.
  typedef double Type;
  /// The length of a SIMD vector of double-precision, floating-point numbers.
  BOOST_STATIC_CONSTEXPR std::size_t Size = 1;
};
#ifdef STLIB_AVX512F
template<>
struct Vector<double, 8> {
  typedef __m512d Type;
  BOOST_STATIC_CONSTEXPR std::size_t Size = 8;
};
#endif
#ifdef __AVX__
template<>
struct Vector<double, 4> {
  typedef __m256d Type;
  BOOST_STATIC_CONSTEXPR std::size_t Size = 4;
};
#endif
#ifdef __SSE__
template<>
struct Vector<double, 2> {
  typedef __m128d Type;
  BOOST_STATIC_CONSTEXPR std::size_t Size = 2;
};
#endif


/// Specialization for unsigned.
template<>
struct Vector<unsigned, 1> {
  /// The SIMD short vector type.
  typedef unsigned Type;
  /// The length of a SIMD vector.
  BOOST_STATIC_CONSTEXPR std::size_t Size = 1;
};
#ifdef STLIB_AVX512F
template<>
struct Vector<unsigned, 16> {
  typedef __m512i Type;
  BOOST_STATIC_CONSTEXPR std::size_t Size = 16;
};
#endif
#ifdef __AVX2__
template<>
struct Vector<unsigned, 8> {
  typedef __m256i Type;
  BOOST_STATIC_CONSTEXPR std::size_t Size = 8;
};
#endif
#ifdef __SSE2__
template<>
struct Vector<unsigned, 4> {
  typedef __m128i Type;
  BOOST_STATIC_CONSTEXPR std::size_t Size = 4;
};
#endif


}
}

#endif // stlib_simd_constants_h
