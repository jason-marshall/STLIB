// -*- C++ -*-

#include "stlib/simd/functions.h"

#include <cassert>

using namespace stlib;

int
main()
{

  assert(simd::conditional(true, 2, 3) == 2);
  assert(simd::conditional(false, 2, 3) == 3);

  {
    __m128 a = _mm_set_ps(4, 3, 2, 1);
    __m128 b = _mm_set_ps(7, 5, 3, 2);
    __m128 c = simd::dot(a, b);
    assert(*reinterpret_cast<const float*>(&c) == 51);
  }

  return 0;
}
