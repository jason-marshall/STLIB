// -*- C++ -*-

#include "stlib/simd/operators.h"

#include <iostream>

#include <array>
#include <vector>
#include <limits>

#include <cassert>
#include <cmath>

struct A {
  __m128 x;
};

int
main()
{

  // aggregate constructor.
  {
    __m128 a = {0.f, 1.f, 2.f, 3.f};
    const float* p = reinterpret_cast<const float*>(&a);
    for (std::size_t i = 0; i != 4; ++i) {
      assert(p[i] == i);
    }
  }
  // setzero
  {
    __m128 a = _mm_setzero_ps();
    const float* p = reinterpret_cast<const float*>(&a);
    for (std::size_t i = 0; i != 4; ++i) {
      assert(p[i] == 0);
    }
    //std::cout << p[0] << '\n';
  }
  // Packed arithmetic.
  {
    const float Eps = 10 * std::numeric_limits<float>::epsilon();
    const __m128 a = {0.f, 1.f, 2.f, 3.f};
    const __m128 b = {2.f, 3.f, 5.f, 7.f};
    __m128 c;
    const float* x = reinterpret_cast<const float*>(&a);
    const float* y = reinterpret_cast<const float*>(&b);
    const float* z = reinterpret_cast<const float*>(&c);
    // add
    c = a + b;
    for (std::size_t i = 0; i != 4; ++i) {
      assert(z[i] == x[i] + y[i]);
    }
    // subtract
    c = a - b;
    for (std::size_t i = 0; i != 4; ++i) {
      assert(z[i] == x[i] - y[i]);
    }
    // multiply
    c = a * b;
    for (std::size_t i = 0; i != 4; ++i) {
      assert(z[i] == x[i] * y[i]);
    }
    // divide
    c = a / b;
    for (std::size_t i = 0; i != 4; ++i) {
      assert(std::abs(z[i] - x[i] / y[i]) < Eps);
    }
    // square root
    c = _mm_sqrt_ps(a);
    for (std::size_t i = 0; i != 4; ++i) {
      assert(std::abs(z[i] - std::sqrt(x[i])) < Eps);
    }
    // minimum
    c = _mm_min_ps(a, b);
    for (std::size_t i = 0; i != 4; ++i) {
      assert(z[i] == std::min(x[i], y[i]));
    }
  }

  {
    std::array<__m128, 1> a;
    a[0] = _mm_set1_ps(0);
  }

  // Cannot explicitly call default constructor.
#if 0
  {
    __m128 a = __m128();
  }
#endif
  // New and delete.
  {
    __m128* a = new __m128[4];
    delete[] a;
  }
  {
    // The following will work with GCC 4.2, but not 4.1 or 4.5.
#if 0
    std::vector<__m128> a;
    __m128 x;
    a.push_back(x);
    std::vector<__m128> b(4, x);
    a.resize(4, x);
#endif
#if 0
    // You can't resize or use the size constructor without an assignment
    // value.
    a.resize(4);
    std::vector<__m128> b(4);
#endif
  }
#if 0
  // This won't work (on 32-bit Ubuntu for example).
  {
    std::vector<A> a(4);
  }
#endif

  return 0;
}
