// -*- C++ -*-

#if !defined(stlib_simd_functions_h)
#define stlib_simd_functions_h

#include "stlib/simd/constants.h"
#include "stlib/simd/operators.h"

#include <algorithm>
#include <cmath>

namespace stlib
{
namespace simd
{


//---------------------------------------------------------------------------
// setzero()
//---------------------------------------------------------------------------

/// Return a vector with all elements set to zero.
template<typename _Float>
typename Vector<_Float>::Type
setzero();

#ifdef STLIB_AVX512F

template<>
inline
__m512
setzero<float>()
{
  return _mm512_setzero_ps();
}
/// Return a vector with all elements set to zero.
template<>
inline
__m512d
setzero<double>()
{
  return _mm512_setzero_pd();
}

#elif defined(__AVX__)

template<>
inline
__m256
setzero<float>()
{
  return _mm256_setzero_ps();
}
template<>
inline
__m256d
setzero<double>()
{
  return _mm256_setzero_pd();
}

#elif defined(__SSE__)

template<>
inline
__m128
setzero<float>()
{
  return _mm_setzero_ps();
}
template<>
inline
__m128d
setzero<double>()
{
  return _mm_setzero_pd();
}

#else

template<>
inline
float
setzero<float>()
{
  return 0;
}
template<>
inline
double
setzero<double>()
{
  return 0;
}

#endif


//---------------------------------------------------------------------------
// setBits()
//---------------------------------------------------------------------------

/// Return a vector with all bits set.
template<typename _Float>
typename Vector<_Float>::Type
setBits();

#ifdef STLIB_AVX512F

template<>
inline
__m512
setBits<float>()
{
  return _mm512_castsi512_ps(_mm512_set1_epi32(-1));
}
/// Return a vector with all elements set to zero.
template<>
inline
__m512d
setBits<double>()
{
  return _mm512_castsi512_pd(_mm512_set1_epi32(-1));
}

#elif defined(__AVX__)

template<>
inline
__m256
setBits<float>()
{
  return _mm256_castsi256_ps(_mm256_set1_epi32(-1));
}
template<>
inline
__m256d
setBits<double>()
{
  return _mm256_castsi256_pd(_mm256_set1_epi32(-1));
}

#elif defined(__SSE__)

template<>
inline
__m128
setBits<float>()
{
  return _mm_castsi128_ps(_mm_set1_epi32(-1));
}
template<>
inline
__m128d
setBits<double>()
{
  return _mm_castsi128_pd(_mm_set1_epi32(-1));
}

#else

template<>
inline
float
setBits<float>()
{
  static_assert(sizeof(std::int32_t) == sizeof(float),
                "Unsupported size for float.");
  std::int32_t constexpr ones = -1;
  return *reinterpret_cast<float const*>(&ones);
}
template<>
inline
double
setBits<double>()
{
  static_assert(sizeof(std::int64_t) == sizeof(double),
                "Unsupported size for double.");
  std::int64_t constexpr ones = -1;
  return *reinterpret_cast<double const*>(&ones);
}

#endif


//---------------------------------------------------------------------------
// set1()
//---------------------------------------------------------------------------
#ifdef STLIB_AVX512F

/// Broadcast floating-point argument to all elements of the result.
static inline
__m512
set1(const float a)
{
  return _mm512_set1_ps(a);
}
/// Broadcast floating-point argument to all elements of the result.
static inline
__m512d
set1(const double a)
{
  return _mm512_set1_pd(a);
}

#elif defined(__AVX__)

/// Broadcast floating-point argument to all elements of the result.
static inline
__m256
set1(const float a)
{
  return _mm256_set1_ps(a);
}
/// Broadcast floating-point argument to all elements of the result.
static inline
__m256d
set1(const double a)
{
  return _mm256_set1_pd(a);
}

#elif defined(__SSE__)

static inline
__m128
set1(const float a)
{
  return _mm_set1_ps(a);
}
static inline
__m128d
set1(const double a)
{
  return _mm_set1_pd(a);
}

#else

static inline
float
set1(const float a)
{
  return a;
}
static inline
double
set1(const double a)
{
  return a;
}

#endif


//---------------------------------------------------------------------------
// load()
//---------------------------------------------------------------------------
#ifdef STLIB_AVX512F

/// Load aligned floating-point elements from memory into result.
static inline
__m512
load(const float* a)
{
  return _mm512_load_ps(a);
}
/// Load aligned floating-point elements from memory into result.
static inline
__m512d
load(const double* a)
{
  return _mm512_load_pd(a);
}

#elif defined(__AVX__)

/// Load aligned floating-point elements from memory into result.
static inline
__m256
load(const float* a)
{
  return _mm256_load_ps(a);
}
/// Load aligned floating-point elements from memory into result.
static inline
__m256d
load(const double* a)
{
  return _mm256_load_pd(a);
}

#elif defined(__SSE__)

static inline
__m128
load(const float* a)
{
  return _mm_load_ps(a);
}
static inline
__m128d
load(const double* a)
{
  return _mm_load_pd(a);
}

#else

static inline
float
load(const float* a)
{
  return *a;
}
static inline
double
load(const double* a)
{
  return *a;
}

#endif


//---------------------------------------------------------------------------
// store()
//---------------------------------------------------------------------------
#ifdef STLIB_AVX512F

/// Store a vector into aligned memory.
static inline
void
store(float* address, const __m512 a)
{
  _mm512_store_ps(address, a);
}
/// Store a vector into aligned memory.
static inline
void
store(double* address, const __m512d a)
{
  _mm512_store_pd(address, a);
}

#endif

#ifdef __AVX__

/// Store a vector into aligned memory.
static inline
void
store(float* address, const __m256 a)
{
  _mm256_store_ps(address, a);
}
/// Store a vector into aligned memory.
static inline
void
store(double* address, const __m256d a)
{
  _mm256_store_pd(address, a);
}

#endif

#ifdef __SSE__

/// Store a vector into aligned memory.
static inline
void
store(float* address, const __m128 a)
{
  _mm_store_ps(address, a);
}
/// Store a vector into aligned memory.
static inline
void
store(double* address, const __m128d a)
{
  _mm_store_pd(address, a);
}

#endif

/// Store a vector into aligned memory.
static inline
void
store(float* address, const float a)
{
  *address = a;
}
/// Store a vector into aligned memory.
static inline
void
store(double* address, const double a)
{
  *address = a;
}


//---------------------------------------------------------------------------
// front()
//---------------------------------------------------------------------------
#ifdef STLIB_AVX512F

/// Return the value at the front (least significant end) of the vector.
static inline
float
front(const __m512 a)
{
  return *reinterpret_cast<const float*>(&a);
}
/// Return the value at the front (least significant end) of the vector.
static inline
double
front(const __m512d a)
{
  return *reinterpret_cast<const double*>(&a);
}

#endif

#ifdef __AVX__

/// Return the value at the front (least significant end) of the vector.
static inline
float
front(const __m256 a)
{
  return *reinterpret_cast<const float*>(&a);
}
/// Return the value at the front (least significant end) of the vector.
static inline
double
front(const __m256d a)
{
  return *reinterpret_cast<const double*>(&a);
}

#endif

#ifdef __SSE__

/// Return the value at the front (least significant end) of the vector.
static inline
float
front(const __m128 a)
{
  return *reinterpret_cast<const float*>(&a);
}
/// Return the value at the front (least significant end) of the vector.
static inline
double
front(const __m128d a)
{
  return *reinterpret_cast<const double*>(&a);
}

#endif

/// Return the value at the front (least significant end) of the vector.
static inline
float
front(const float a)
{
  return a;
}
/// Return the value at the front (least significant end) of the vector.
static inline
double
front(const double a)
{
  return a;
}



//---------------------------------------------------------------------------
// sum()
//---------------------------------------------------------------------------
#ifdef STLIB_AVX512F

/// Return the horizontal sum of the elements.
static inline
float
sum(const __m512 a)
{
  __m256 x = _mm512_castps512_ps256(a);
  __m256 y = *(reinterpret_cast<const __m256*>(&a) + 1);
  // a0+a1, a2+a3, a8+a9, a10+a11, a4+a5, a6+a7, a12+a13, a14+a15
  x = _mm256_hadd_ps(x, y);
  // a0+a1+a2+a3, a8+a9+a10+a11, a0+a1+a2+a3, a8+a9+a10+a11,
  // a4+a5+a6+a7, a12+a13+a14+a15, a4+a5+a6+a7, a12+a13+a14+a15
  x = _mm256_hadd_ps(x, x);
  // a0+a1+a2+a3+a8+a9+a10+a11, a0+a1+a2+a3+a8+a9+a10+a11,
  // a0+a1+a2+a3+a8+a9+a10+a11, a0+a1+a2+a3+a8+a9+a10+a11,
  // a4+a5+a6+a7+a12+a13+a14+a15, a4+a5+a6+a7+a12+a13+a14+a15
  // a4+a5+a6+a7+a12+a13+a14+a15, a4+a5+a6+a7+a12+a13+a14+a15
  x = _mm256_hadd_ps(x, x);
  const float* p = reinterpret_cast<const float*>(&x);
  return p[0] + p[4];
}
/// Return the horizontal sum of the elements.
static inline
double
sum(const __m512d a)
{
  __m256d x = _mm512_castpd512_pd256(a);
  __m256d y = *(reinterpret_cast<const __m256d*>(&a) + 1);
  // a0+a1, a4+a5, a2+a3, a6+a7
  x = _mm256_hadd_pd(x, y);
  // a0+a1+a4+a5, a0+a1+a4+a5, a2+a3+a6+a7, a2+a3+a6+a7
  x = _mm256_hadd_pd(x, x);
  const double* p = reinterpret_cast<const double*>(&x);
  return p[0] + p[2];
}

#endif

#ifdef __AVX__

static inline
float
sum(__m256 a)
{
  // a0+a1, a2+a3, a0+a1, a2+a3, a4+a5, a6+a7, a4+a5, a6+a7
  a = _mm256_hadd_ps(a, a);
  // a0+a1+a2+a3, a0+a1+a2+a3, a0+a1+a2+a3, a0+a1+a2+a3,
  // a4+a5+a6+a7, a4+a5+a6+a7, a4+a5+a6+a7, a4+a5+a6+a7
  a = _mm256_hadd_ps(a, a);
  const float* p = reinterpret_cast<const float*>(&a);
  return p[0] + p[4];
}
static inline
double
sum(__m256d a)
{
  // a0+a1, a0+a1, a2+a3, a2+a3
  a = _mm256_hadd_pd(a, a);
  const double* p = reinterpret_cast<const double*>(&a);
  return p[0] + p[2];
}

#endif

#ifdef __SSE3__

static inline
float
sum(__m128 a)
{
  // a0+a1, a2+a3, a0+a1, a2+a3
  a = _mm_hadd_ps(a, a);
  const float* p = reinterpret_cast<const float*>(&a);
  return p[0] + p[1];
}

#elif defined(__SSE__)

static inline
float
sum(__m128 a)
{
  const float* p = reinterpret_cast<const float*>(&a);
  return p[0] + p[1] + p[2] + p[3];
}

#endif

#ifdef __SSE__

static inline
double
sum(const __m128d a)
{
  const double* p = reinterpret_cast<const double*>(&a);
  return p[0] + p[1];
}

#endif

static inline
float
sum(const float a)
{
  return a;
}
static inline
double
sum(const double a)
{
  return a;
}




//---------------------------------------------------------------------------
// bitwiseAnd()
//---------------------------------------------------------------------------
#ifdef STLIB_AVX512F

/// Return the value at the front (least significant end) of the vector.
static inline
__m512
bitwiseAnd(const __m512 a, const __m512 b)
{
  return _mm512_and_ps(a, b);
}
/// Return the value at the front (least significant end) of the vector.
static inline
__m512d
bitwiseAnd(const __m512d a, const __m512d b)
{
  return _mm512_and_pd(a, b);
}

#endif

#ifdef __AVX__

static inline
__m256
bitwiseAnd(const __m256 a, const __m256 b)
{
  return _mm256_and_ps(a, b);
}
static inline
__m256d
bitwiseAnd(const __m256d a, const __m256d b)
{
  return _mm256_and_pd(a, b);
}

#endif

#ifdef __SSE__

static inline
__m128
bitwiseAnd(const __m128 a, const __m128 b)
{
  return _mm_and_ps(a, b);
}
static inline
__m128d
bitwiseAnd(const __m128d a, const __m128d b)
{
  return _mm_and_pd(a, b);
}

#endif

static inline
float
bitwiseAnd(const float a, const float b)
{
  float result;
  const unsigned char* x = reinterpret_cast<const unsigned char*>(&a);
  const unsigned char* y = reinterpret_cast<const unsigned char*>(&b);
  unsigned char* r = reinterpret_cast<unsigned char*>(&result);
  for (std::size_t i = 0; i != sizeof(float); ++i) {
    *r = *x & *y;
  }
  return result;
}
static inline
double
bitwiseAnd(const double a, const double b)
{
  double result;
  const unsigned char* x = reinterpret_cast<const unsigned char*>(&a);
  const unsigned char* y = reinterpret_cast<const unsigned char*>(&b);
  unsigned char* r = reinterpret_cast<unsigned char*>(&result);
  for (std::size_t i = 0; i != sizeof(double); ++i) {
    *r = *x & *y;
  }
  return result;
}



//---------------------------------------------------------------------------
// bitwiseOr()
//---------------------------------------------------------------------------
#ifdef STLIB_AVX512F

/// Return the value at the front (least significant end) of the vector.
static inline
__m512
bitwiseOr(const __m512 a, const __m512 b)
{
  return _mm512_or_ps(a, b);
}
/// Return the value at the front (least significant end) of the vector.
static inline
__m512d
bitwiseOr(const __m512d a, const __m512d b)
{
  return _mm512_or_pd(a, b);
}

#endif

#ifdef __AVX__

static inline
__m256
bitwiseOr(const __m256 a, const __m256 b)
{
  return _mm256_or_ps(a, b);
}
static inline
__m256d
bitwiseOr(const __m256d a, const __m256d b)
{
  return _mm256_or_pd(a, b);
}

#endif

#ifdef __SSE__

static inline
__m128
bitwiseOr(const __m128 a, const __m128 b)
{
  return _mm_or_ps(a, b);
}
static inline
__m128d
bitwiseOr(const __m128d a, const __m128d b)
{
  return _mm_or_pd(a, b);
}

#endif

static inline
float
bitwiseOr(const float a, const float b)
{
  float result;
  const unsigned char* x = reinterpret_cast<const unsigned char*>(&a);
  const unsigned char* y = reinterpret_cast<const unsigned char*>(&b);
  unsigned char* r = reinterpret_cast<unsigned char*>(&result);
  for (std::size_t i = 0; i != sizeof(float); ++i) {
    *r = *x | *y;
  }
  return result;
}
static inline
double
bitwiseOr(const double a, const double b)
{
  double result;
  const unsigned char* x = reinterpret_cast<const unsigned char*>(&a);
  const unsigned char* y = reinterpret_cast<const unsigned char*>(&b);
  unsigned char* r = reinterpret_cast<unsigned char*>(&result);
  for (std::size_t i = 0; i != sizeof(double); ++i) {
    *r = *x | *y;
  }
  return result;
}




//---------------------------------------------------------------------------
// min()
//---------------------------------------------------------------------------
#ifdef STLIB_AVX512F

/// Store the minimum values in the result.
/** \note This function has odd behavior for NaN's. If either argument
  is a NaN, the second argument is returned. */
static inline
__m512
min(const __m512 a, const __m512 b)
{
  return _mm512_min_ps(a, b);
}
/// Store the minimum values in the result.
static inline
__m512d
min(const __m512d a, const __m512d b)
{
  return _mm512_min_pd(a, b);
}

#endif

#ifdef __AVX__

static inline
__m256
min(const __m256 a, const __m256 b)
{
  return _mm256_min_ps(a, b);
}
static inline
__m256d
min(const __m256d a, const __m256d b)
{
  return _mm256_min_pd(a, b);
}

#endif

#ifdef __SSE__

static inline
__m128
min(const __m128 a, const __m128 b)
{
  return _mm_min_ps(a, b);
}
static inline
__m128d
min(const __m128d a, const __m128d b)
{
  return _mm_min_pd(a, b);
}

#endif

static inline
float
min(const float a, const float b)
{
  return std::min(a, b);
}
static inline
double
min(const double a, const double b)
{
  return std::min(a, b);
}




//---------------------------------------------------------------------------
// max()
//---------------------------------------------------------------------------
#ifdef STLIB_AVX512F

/// Store the maximum values in the result.
static inline
__m512
max(const __m512 a, const __m512 b)
{
  return _mm512_max_ps(a, b);
}
/// Store the maximum values in the result.
static inline
__m512d
max(const __m512d a, const __m512d b)
{
  return _mm512_max_pd(a, b);
}

#endif

#ifdef __AVX__

static inline
__m256
max(const __m256 a, const __m256 b)
{
  return _mm256_max_ps(a, b);
}
static inline
__m256d
max(const __m256d a, const __m256d b)
{
  return _mm256_max_pd(a, b);
}

#endif

#ifdef __SSE__

static inline
__m128
max(const __m128 a, const __m128 b)
{
  return _mm_max_ps(a, b);
}
static inline
__m128d
max(const __m128d a, const __m128d b)
{
  return _mm_max_pd(a, b);
}

#endif

static inline
float
max(const float a, const float b)
{
  return std::max(a, b);
}
static inline
double
max(const double a, const double b)
{
  return std::max(a, b);
}




//---------------------------------------------------------------------------
// min() - horizontal
//---------------------------------------------------------------------------
#ifdef STLIB_AVX512F

/// Return the horizontal minimum.
static inline
float
min(__m512 const a)
{
  float const* const p = reinterpret_cast<float const*>(&a);
  float result = *p;
  for (std::size_t i = 1; i != 16; ++i) {
    result = std::min(result, p[i]);
  }
  return result;
}
/// Return the horizontal minimum.
static inline
double
min(__m512d const a)
{
  double const* const p = reinterpret_cast<double const*>(&a);
  double result = *p;
  for (std::size_t i = 1; i != 8; ++i) {
    result = std::min(result, p[i]);
  }
  return result;
}

#endif

#ifdef __AVX__

static inline
float
min(__m256 const a)
{
  float const* const p = reinterpret_cast<float const*>(&a);
  float result = *p;
  for (std::size_t i = 1; i != 8; ++i) {
    result = std::min(result, p[i]);
  }
  return result;
}
static inline
double
min(__m256d const a)
{
  double const* const p = reinterpret_cast<double const*>(&a);
  double result = *p;
  for (std::size_t i = 1; i != 4; ++i) {
    result = std::min(result, p[i]);
  }
  return result;
}

#endif

#ifdef __SSE__

static inline
float
min(__m128 const a)
{
  float const* const p = reinterpret_cast<float const*>(&a);
  float result = *p;
  for (std::size_t i = 1; i != 4; ++i) {
    result = std::min(result, p[i]);
  }
  return result;
}
static inline
double
min(__m128d const a)
{
  double const* const p = reinterpret_cast<double const*>(&a);
  return std::min(p[0], p[1]);
}

#endif

static inline
float
min(float const a)
{
  return a;
}
static inline
double
min(double const a)
{
  return a;
}




//---------------------------------------------------------------------------
// max() - horizontal
//---------------------------------------------------------------------------
#ifdef STLIB_AVX512F

/// Return the horizontal maximum.
static inline
float
max(__m512 const a)
{
  float const* const p = reinterpret_cast<float const*>(&a);
  float result = *p;
  for (std::size_t i = 1; i != 16; ++i) {
    result = std::max(result, p[i]);
  }
  return result;
}
/// Return the horizontal maximum.
static inline
double
max(__m512d const a)
{
  double const* const p = reinterpret_cast<double const*>(&a);
  double result = *p;
  for (std::size_t i = 1; i != 8; ++i) {
    result = std::max(result, p[i]);
  }
  return result;
}

#endif

#ifdef __AVX__

static inline
float
max(__m256 const a)
{
  float const* const p = reinterpret_cast<float const*>(&a);
  float result = *p;
  for (std::size_t i = 1; i != 8; ++i) {
    result = std::max(result, p[i]);
  }
  return result;
}
static inline
double
max(__m256d const a)
{
  double const* const p = reinterpret_cast<double const*>(&a);
  double result = *p;
  for (std::size_t i = 1; i != 4; ++i) {
    result = std::max(result, p[i]);
  }
  return result;
}

#endif

#ifdef __SSE__

static inline
float
max(__m128 const a)
{
  float const* const p = reinterpret_cast<float const*>(&a);
  float result = *p;
  for (std::size_t i = 1; i != 4; ++i) {
    result = std::max(result, p[i]);
  }
  return result;
}
static inline
double
max(__m128d const a)
{
  double const* const p = reinterpret_cast<double const*>(&a);
  return std::max(p[0], p[1]);
}

#endif

static inline
float
max(float const a)
{
  return a;
}
static inline
double
max(double const a)
{
  return a;
}




//---------------------------------------------------------------------------
// equal()
//---------------------------------------------------------------------------
#ifdef STLIB_AVX512F

/// Equality comparison.
static inline
__mmask16
equal(const __m512 a, const __m512 b)
{
  return _mm512_cmp_ps_mask(a, b, _CMP_EQ_OQ);
}
/// Equality comparison.
static inline
__mmask8
equal(const __m512d a, const __m512d b)
{
  return _mm512_cmp_pd_mask(a, b, _CMP_EQ_OQ);
}

#endif

#ifdef __AVX__

static inline
__m256
equal(const __m256 a, const __m256 b)
{
  return _mm256_cmp_ps(a, b, _CMP_EQ_OQ);
}
static inline
__m256d
equal(const __m256d a, const __m256d b)
{
  return _mm256_cmp_pd(a, b, _CMP_EQ_OQ);
}

#endif

#ifdef __SSE__

static inline
__m128
equal(const __m128 a, const __m128 b)
{
  return _mm_cmpeq_ps(a, b);
}
static inline
__m128d
equal(const __m128d a, const __m128d b)
{
  return _mm_cmpeq_pd(a, b);
}

#endif

static inline
float
equal(const float a, const float b)
{
  return a == b;
}
static inline
double
equal(const double a, const double b)
{
  return a == b;
}




//---------------------------------------------------------------------------
// notEqual()
//---------------------------------------------------------------------------
#ifdef STLIB_AVX512F

/// Inequality comparison.
static inline
__mmask16
notEqual(const __m512 a, const __m512 b)
{
  return _mm512_cmp_ps_mask(a, b, _CMP_NEQ_OQ);
}
/// Inequality comparison.
static inline
__mmask8
notEqual(const __m512d a, const __m512d b)
{
  return _mm512_cmp_pd_mask(a, b, _CMP_NEQ_OQ);
}

#endif

#ifdef __AVX__

static inline
__m256
notEqual(const __m256 a, const __m256 b)
{
  return _mm256_cmp_ps(a, b, _CMP_NEQ_OQ);
}
static inline
__m256d
notEqual(const __m256d a, const __m256d b)
{
  return _mm256_cmp_pd(a, b, _CMP_NEQ_OQ);
}

#endif

#ifdef __SSE__

static inline
__m128
notEqual(const __m128 a, const __m128 b)
{
  return _mm_cmpneq_ps(a, b);
}
static inline
__m128d
notEqual(const __m128d a, const __m128d b)
{
  return _mm_cmpneq_pd(a, b);
}

#endif

static inline
float
notEqual(const float a, const float b)
{
  return a != b;
}
static inline
double
notEqual(const double a, const double b)
{
  return a != b;
}




//---------------------------------------------------------------------------
// less()
//---------------------------------------------------------------------------
#ifdef STLIB_AVX512F

/// Less than comparison.
static inline
__mmask16
less(const __m512 a, const __m512 b)
{
  return _mm512_cmp_ps_mask(a, b, _CMP_LT_OQ);
}
/// Less than comparison.
static inline
__mmask8
less(const __m512d a, const __m512d b)
{
  return _mm512_cmp_pd_mask(a, b, _CMP_LT_OQ);
}

#endif

#ifdef __AVX__

static inline
__m256
less(const __m256 a, const __m256 b)
{
  return _mm256_cmp_ps(a, b, _CMP_LT_OQ);
}
static inline
__m256d
less(const __m256d a, const __m256d b)
{
  return _mm256_cmp_pd(a, b, _CMP_LT_OQ);
}

#endif

#ifdef __SSE__

static inline
__m128
less(const __m128 a, const __m128 b)
{
  return _mm_cmplt_ps(a, b);
}
static inline
__m128d
less(const __m128d a, const __m128d b)
{
  return _mm_cmplt_pd(a, b);
}

#endif

static inline
float
less(const float a, const float b)
{
  return a < b;
}
static inline
double
less(const double a, const double b)
{
  return a < b;
}




//---------------------------------------------------------------------------
// lessEqual()
//---------------------------------------------------------------------------
#ifdef STLIB_AVX512F

/// Less than or equal to comparison.
static inline
__mmask16
lessEqual(const __m512 a, const __m512 b)
{
  return _mm512_cmp_ps_mask(a, b, _CMP_LE_OQ);
}
/// Less than or equal to comparison.
static inline
__mmask8
lessEqual(const __m512d a, const __m512d b)
{
  return _mm512_cmp_pd_mask(a, b, _CMP_LE_OQ);
}

#endif

#ifdef __AVX__

static inline
__m256
lessEqual(const __m256 a, const __m256 b)
{
  return _mm256_cmp_ps(a, b, _CMP_LE_OQ);
}
static inline
__m256d
lessEqual(const __m256d a, const __m256d b)
{
  return _mm256_cmp_pd(a, b, _CMP_LE_OQ);
}

#endif

#ifdef __SSE__

static inline
__m128
lessEqual(const __m128 a, const __m128 b)
{
  return _mm_cmple_ps(a, b);
}
static inline
__m128d
lessEqual(const __m128d a, const __m128d b)
{
  return _mm_cmple_pd(a, b);
}

#endif

static inline
float
lessEqual(const float a, const float b)
{
  return a <= b;
}
static inline
double
lessEqual(const double a, const double b)
{
  return a <= b;
}



//---------------------------------------------------------------------------
// greater()
//---------------------------------------------------------------------------
#ifdef STLIB_AVX512F

/// Greater than comparison.
static inline
__mmask16
greater(const __m512 a, const __m512 b)
{
  return _mm512_cmp_ps_mask(a, b, _CMP_GT_OQ);
}
/// Greater than comparison.
static inline
__mmask8
greater(const __m512d a, const __m512d b)
{
  return _mm512_cmp_pd_mask(a, b, _CMP_GT_OQ);
}

#endif

#ifdef __AVX__

static inline
__m256
greater(const __m256 a, const __m256 b)
{
  return _mm256_cmp_ps(a, b, _CMP_GT_OQ);
}
static inline
__m256d
greater(const __m256d a, const __m256d b)
{
  return _mm256_cmp_pd(a, b, _CMP_GT_OQ);
}

#endif

#ifdef __SSE__

static inline
__m128
greater(const __m128 a, const __m128 b)
{
  return _mm_cmpgt_ps(a, b);
}
static inline
__m128d
greater(const __m128d a, const __m128d b)
{
  return _mm_cmpgt_pd(a, b);
}

#endif

static inline
float
greater(const float a, const float b)
{
  return a > b;
}
static inline
double
greater(const double a, const double b)
{
  return a > b;
}




//---------------------------------------------------------------------------
// moveMask()
//---------------------------------------------------------------------------
#ifdef STLIB_AVX512F

// Not yet implemented.
int
moveMask(__m512 a);

// Not yet implemented.
int
moveMask(__m512d a);

#endif

#ifdef __AVX__

static inline
int
moveMask(const __m256 a)
{
  return _mm256_movemask_ps(a);
}
static inline
int
moveMask(const __m256d a)
{
  return _mm256_movemask_pd(a);
}

#endif

#ifdef __SSE__

static inline
int
moveMask(const __m128 a)
{
  return _mm_movemask_ps(a);
}
static inline
int
moveMask(const __m128d a)
{
  return _mm_movemask_pd(a);
}

#endif

static inline
int
moveMask(const float a)
{
  // Get the most significant bit.
  return (*(reinterpret_cast<const unsigned char*>(&a) + sizeof(float) - 1) &
          0x80) >> 7;
}
static inline
int
moveMask(const double a)
{
  return (*(reinterpret_cast<const unsigned char*>(&a) + sizeof(double) - 1) &
          0x80) >> 7;
}




//---------------------------------------------------------------------------
// abs()
//---------------------------------------------------------------------------
#ifdef STLIB_AVX512F

/// Return the absolute value.
static inline
__m512
abs(const __m512 a)
{
#if 0
  // CONTINUE: I don't know why this intrinsic is not available.
  return _mm512_abs_ps(a);
#endif
  static const __m512 SignMask = _mm512_set1_ps(-0.f);
  return _mm512_andnot_ps(SignMask, a);
}
/// Return the absolute value.
static inline
__m512d
abs(const __m512d a)
{
#if 0
  // CONTINUE: I don't know why this intrinsic is not available.
  return _mm512_abs_pd(a);
#endif
  static const __m512d SignMask = _mm512_set1_pd(-0.);
  return _mm512_andnot_pd(SignMask, a);
}

#endif

#ifdef __AVX__

static inline
__m256
abs(const __m256 a)
{
  static const __m256 SignMask = _mm256_set1_ps(-0.f);
  return _mm256_andnot_ps(SignMask, a);
}
static inline
__m256d
abs(const __m256d a)
{
  static const __m256d SignMask = _mm256_set1_pd(-0.);
  return _mm256_andnot_pd(SignMask, a);
}

#endif

#ifdef __SSE__

static inline
__m128
abs(const __m128 a)
{
  static const __m128 SignMask = _mm_set1_ps(-0.f);
  return _mm_andnot_ps(SignMask, a);
}
static inline
__m128d
abs(const __m128d a)
{
  static const __m128d SignMask = _mm_set1_pd(-0.);
  return _mm_andnot_pd(SignMask, a);
}

#endif

static inline
float
abs(const float a)
{
  return std::abs(a);
}
static inline
double
abs(const double a)
{
  return std::abs(a);
}




//---------------------------------------------------------------------------
// sqrt()
//---------------------------------------------------------------------------
#ifdef STLIB_AVX512F

/// Return the square root.
static inline
__m512
sqrt(const __m512 a)
{
  return _mm512_sqrt_ps(a);
}
/// Return the square root.
static inline
__m512d
sqrt(const __m512d a)
{
  return _mm512_sqrt_pd(a);
}

#endif

#ifdef __AVX__

static inline
__m256
sqrt(const __m256 a)
{
  return _mm256_sqrt_ps(a);
}
static inline
__m256d
sqrt(const __m256d a)
{
  return _mm256_sqrt_pd(a);
}

#endif

#ifdef __SSE__

static inline
__m128
sqrt(const __m128 a)
{
  return _mm_sqrt_ps(a);
}
static inline
__m128d
sqrt(const __m128d a)
{
  return _mm_sqrt_pd(a);
}

#endif

static inline
float
sqrt(const float a)
{
  return std::sqrt(a);
}
static inline
double
sqrt(const double a)
{
  return std::sqrt(a);
}




//---------------------------------------------------------------------------
// Scalar
//---------------------------------------------------------------------------


/// If the predicate is true, return \c first, otherwise return \c second.
static inline
float
conditional(const bool predicate, const float first, const float second)
{
  return predicate ? first : second;
}


/// If the predicate is true, return \c first, otherwise return \c second.
static inline
int
conditional(const bool predicate, const int first, const int second)
{
  return predicate ? first : second;
}






//---------------------------------------------------------------------------
// SSE
//---------------------------------------------------------------------------
#ifdef __SSE__

/// For each of the elements, if the predicate is true, return \c first, otherwise return \c second.
static inline
__m128
conditional(const __m128 predicate, const __m128 first, const __m128 second)
{
  return _mm_or_ps(_mm_and_ps(predicate, first),
                   _mm_andnot_ps(predicate, second));
}


/// For each of the elements, if the predicate is true, return \c first, otherwise return \c second.
static inline
__m128d
conditional(const __m128d predicate, const __m128d first, const __m128d second)
{
  return _mm_or_pd(_mm_and_pd(predicate, first),
                   _mm_andnot_pd(predicate, second));
}

#endif // __SSE__


#ifdef __SSE2__

/// For each of the elements, if the predicate is true, return \c first, otherwise return \c second.
static inline
__m128i
conditional(const __m128i predicate, const __m128i first,
            const __m128i second)
{
  return _mm_or_si128(_mm_and_si128(predicate, first),
                      _mm_andnot_si128(predicate, second));
}

#endif // __SSE2__

//---------------------------------------------------------------------------
// AVX
//---------------------------------------------------------------------------
#ifdef __AVX__


/// For each of the elements, if the predicate is true, return \c first, otherwise return \c second.
static inline
__m256
conditional(const __m256 predicate, const __m256 first, const __m256 second)
{
  return _mm256_or_ps(_mm256_and_ps(predicate, first),
                      _mm256_andnot_ps(predicate, second));
}


/// For each of the elements, if the predicate is true, return \c first, otherwise return \c second.
static inline
__m256d
conditional(const __m256d predicate, const __m256d first, const __m256d second)
{
  return _mm256_or_pd(_mm256_and_pd(predicate, first),
                      _mm256_andnot_pd(predicate, second));
}


#endif // __AVX__




//---------------------------------------------------------------------------
// AVX2
//---------------------------------------------------------------------------
#ifdef __AVX2__

/// For each of the elements, if the predicate is true, return \c first, otherwise return \c second.
static inline
__m256i
conditional(const __m256i predicate, const __m256i first,
            const __m256i second)
{
  return _mm256_or_si256(_mm256_and_si256(predicate, first),
                         _mm256_andnot_si256(predicate, second));
}

#endif



//---------------------------------------------------------------------------
// Dot product.
//---------------------------------------------------------------------------
#ifdef __SSE4_1__
/// Compute the dot product, return the result in the low float.
/** The values in the other elements are undefined. */
static inline
__m128
dot(const __m128 a, const __m128 b)
{
  return _mm_dp_ps(a, b, 0xF1);
}
#elif defined(__SSE3__)
static inline
__m128
dot(__m128 a, const __m128 b)
{
  // If the dot product is not available, use two horizontal
  // additions.
  a = a * b;
  a = _mm_hadd_ps(a, a);
  return _mm_hadd_ps(a, a);
}
#elif defined(__SSE__)
static inline
__m128
dot(__m128 a, const __m128 b)
{
  a = a * b;
  float* x = reinterpret_cast<float*>(&a);
  x[0] += x[1] + x[2] + x[3];
  return a;
}
#endif


} // namespace simd
} // namespace stlib

#endif
