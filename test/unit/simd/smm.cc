// -*- C++ -*-

#ifdef __SSE4_1__
#include <smmintrin.h>
#endif

#include <iostream>

#include <cassert>
#include <cmath>

int
main()
{

#ifdef __SSE4_1__
  {
    const __m128 a = {0.f, 1.f, 2.f, 3.f};
    const __m128 b = {2.f, 3.f, 5.f, 7.f};
    __m128 c;
    const float* x = reinterpret_cast<const float*>(&a);
    const float* y = reinterpret_cast<const float*>(&b);
    const float* z = reinterpret_cast<const float*>(&c);
    // dot
    c = _mm_dp_ps(a, b, 0xf1);
    assert(z[0] == x[0] * y[0] + x[1] * y[1] + x[2] * y[2] + x[3] * y[3]);
    assert(z[1] == 0);
    assert(z[2] == 0);
    assert(z[3] == 0);
  }
#else
  std::cout << "SSE4.1 is not supported.\n";
#endif

  return 0;
}
