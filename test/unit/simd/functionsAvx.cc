// -*- C++ -*-

#include "stlib/simd/functions.h"

#include <cassert>


using namespace stlib;

int
main()
{
#ifdef __AVX2__
  {
    float zero = 0;
    float one;
    * reinterpret_cast<unsigned*>(&one) = 0xFFFFFFFF;
    __m256 r = simd::conditional(_mm256_set_ps(one, one, one, one,
    one, one, one, one),
    _mm_set1_ps(2), _mm_set1_ps(3));
    const float* p = reinterpret_cast<const float*>(&r);
    assert(p[0] == 2);
    assert(p[1] == 2);
    assert(p[2] == 2);
    assert(p[3] == 2);
    assert(p[4] == 2);
    assert(p[5] == 2);
    assert(p[6] == 2);
    assert(p[7] == 2);

    r = simd::conditional(_mm256_set_ps(zero, zero, zero, zero,
    zero, zero, zero, zero),
    _mm_set1_ps(2), _mm_set1_ps(3));
    assert(p[0] == 3);
    assert(p[1] == 3);
    assert(p[2] == 3);
    assert(p[3] == 3);
    assert(p[4] == 3);
    assert(p[5] == 3);
    assert(p[6] == 3);
    assert(p[7] == 3);

    r = simd::conditional(_mm256_set_ps(zero, zero, zero, zero,
    zero, zero, zero, one),
    _mm_set1_ps(2), _mm_set1_ps(3));
    assert(p[0] == 2);
    assert(p[1] == 3);
    assert(p[2] == 3);
    assert(p[3] == 3);
    assert(p[4] == 3);
    assert(p[5] == 3);
    assert(p[6] == 3);
    assert(p[7] == 3);

    r = simd::conditional(_mm256_set_ps(one, zero, zero, zero,
    zero, zero, zero, zero),
    _mm_set1_ps(2), _mm_set1_ps(3));
    assert(p[0] == 3);
    assert(p[1] == 3);
    assert(p[2] == 3);
    assert(p[3] == 3);
    assert(p[4] == 3);
    assert(p[5] == 3);
    assert(p[6] == 3);
    assert(p[7] == 2);
  }
#endif

  return 0;
}
