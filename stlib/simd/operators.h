// -*- C++ -*-

#ifndef stlib_simd_operators_h
#define stlib_simd_operators_h

#include "stlib/simd/macros.h"

#include <emmintrin.h>

// A GCC extension defines arithmetic operators for SIMD vector types.
// Here we define them for other compilers.
#if (defined(__INTEL_COMPILER) || defined(_MSC_VER))

#ifdef __SSE__

//
// Single-precision.
//

static inline
__m128
operator+(__m128 const a)
{
  return a;
}

static inline
__m128
operator-(__m128 const a)
{
  // Flip the sign bits.
  return _mm_xor_ps(a, _mm_set1_ps(-0.f));
}

static inline
__m128
operator+(__m128 const a, __m128 const b)
{
  return _mm_add_ps(a, b);
}

static inline
__m128
operator-(__m128 const a, __m128 const b)
{
  return _mm_sub_ps(a, b);
}

static inline
__m128
operator*(__m128 const a, __m128 const b)
{
  return _mm_mul_ps(a, b);
}

static inline
__m128
operator/(__m128 const a, __m128 const b)
{
  return _mm_div_ps(a, b);
}

static inline
__m128&
operator+=(__m128& a, __m128 const b)
{
  a = a + b;
  return a;
}

static inline
__m128&
operator-=(__m128& a, __m128 const b)
{
  a = a - b;
  return a;
}

static inline
__m128&
operator*=(__m128& a, __m128 const b)
{
  a = a * b;
  return a;
}

static inline
__m128&
operator/=(__m128& a, __m128 const b)
{
  a = a / b;
  return a;
}

//
// Double-precision.
//

static inline
__m128d
operator+(__m128d const a)
{
  return a;
}

static inline
__m128d
operator-(__m128d const a)
{
  // Flip the sign bits.
  return _mm_xor_pd(a, _mm_set1_pd(-0.));
}

static inline
__m128d
operator+(__m128d const a, __m128d const b)
{
  return _mm_add_pd(a, b);
}

static inline
__m128d
operator-(__m128d const a, __m128d const b)
{
  return _mm_sub_pd(a, b);
}

static inline
__m128d
operator*(__m128d const a, __m128d const b)
{
  return _mm_mul_pd(a, b);
}

static inline
__m128d
operator/(__m128d const a, __m128d const b)
{
  return _mm_div_pd(a, b);
}

static inline
__m128d&
operator+=(__m128d& a, __m128d const b)
{
  a = a + b;
  return a;
}

static inline
__m128d&
operator-=(__m128d& a, __m128d const b)
{
  a = a - b;
  return a;
}

static inline
__m128d&
operator*=(__m128d& a, __m128d const b)
{
  a = a * b;
  return a;
}

static inline
__m128d&
operator/=(__m128d& a, __m128d const b)
{
  a = a / b;
  return a;
}

#endif // __SSE__

#ifdef __AVX__

//
// Single-precision.
//

static inline
__m256
operator+(__m256 const a)
{
  return a;
}

static inline
__m256
operator-(__m256 const a)
{
  // Flip the sign bits.
  return _mm256_xor_ps(a, _mm256_set1_ps(-0.f));
}

static inline
__m256
operator+(__m256 const a, __m256 const b)
{
  return _mm256_add_ps(a, b);
}

static inline
__m256
operator-(__m256 const a, __m256 const b)
{
  return _mm256_sub_ps(a, b);
}

static inline
__m256
operator*(__m256 const a, __m256 const b)
{
  return _mm256_mul_ps(a, b);
}

static inline
__m256
operator/(__m256 const a, __m256 const b)
{
  return _mm256_div_ps(a, b);
}

static inline
__m256&
operator+=(__m256& a, __m256 const b)
{
  a = a + b;
  return a;
}

static inline
__m256&
operator-=(__m256& a, __m256 const b)
{
  a = a - b;
  return a;
}

static inline
__m256&
operator*=(__m256& a, __m256 const b)
{
  a = a * b;
  return a;
}

static inline
__m256&
operator/=(__m256& a, __m256 const b)
{
  a = a / b;
  return a;
}

//
// Double-precision.
//

static inline
__m256d
operator+(__m256d const a)
{
  return a;
}

static inline
__m256d
operator-(__m256d const a)
{
  // Flip the sign bits.
  return _mm256_xor_pd(a, _mm256_set1_pd(-0.));
}

static inline
__m256d
operator+(__m256d const a, __m256d const b)
{
  return _mm256_add_pd(a, b);
}

static inline
__m256d
operator-(__m256d const a, __m256d const b)
{
  return _mm256_sub_pd(a, b);
}

static inline
__m256d
operator*(__m256d const a, __m256d const b)
{
  return _mm256_mul_pd(a, b);
}

static inline
__m256d
operator/(__m256d const a, __m256d const b)
{
  return _mm256_div_pd(a, b);
}

static inline
__m256d&
operator+=(__m256d& a, __m256d const b)
{
  a = a + b;
  return a;
}

static inline
__m256d&
operator-=(__m256d& a, __m256d const b)
{
  a = a - b;
  return a;
}

static inline
__m256d&
operator*=(__m256d& a, __m256d const b)
{
  a = a * b;
  return a;
}

static inline
__m256d&
operator/=(__m256d& a, __m256d const b)
{
  a = a / b;
  return a;
}

#endif // __AVX__

#endif // (defined(__INTEL_COMPILER) || defined(_MSC_VER))

#endif
