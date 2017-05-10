// -*- C++ -*-

#include "stlib/simd/functions.h"

#include <cassert>

using namespace stlib;

int
main()
{
#ifdef __SSE__
  float zero = 0;
  float one;
  *reinterpret_cast<unsigned*>(&one) = 0xFFFFFFFF;
  __m128 r = simd::conditional(_mm_set_ps(one, one, one, one), _mm_set1_ps(2),
                               _mm_set1_ps(3));
  const float* p = reinterpret_cast<const float*>(&r);
  assert(p[0] == 2);
  assert(p[1] == 2);
  assert(p[2] == 2);
  assert(p[3] == 2);

  r = simd::conditional(_mm_set_ps(zero, zero, zero, one), _mm_set1_ps(2),
                        _mm_set1_ps(3));
  assert(p[0] == 2);
  assert(p[1] == 3);
  assert(p[2] == 3);
  assert(p[3] == 3);

  r = simd::conditional(_mm_set_ps(one, zero, zero, zero), _mm_set1_ps(2),
                        _mm_set1_ps(3));
  assert(p[0] == 3);
  assert(p[1] == 3);
  assert(p[2] == 3);
  assert(p[3] == 2);

  r = simd::conditional(_mm_set_ps(zero, zero, zero, zero), _mm_set1_ps(2),
                        _mm_set1_ps(3));
  assert(p[0] == 3);
  assert(p[1] == 3);
  assert(p[2] == 3);
  assert(p[3] == 3);

  {
    __m128 a = _mm_set_ps(4, 3, 2, 1);
    __m128 b = _mm_set_ps(7, 5, 3, 2);
    __m128 c = simd::dot(a, b);
    assert(*reinterpret_cast<const float*>(&c) == 51);
  }
#endif

  return 0;
}
