// -*- C++ -*-

#include "stlib/ads/timer.h"
#include "stlib/ext/array.h"
#include "stlib/simd/operators.h"

#include <iostream>

using namespace stlib;

int
main()
{

  ads::Timer timer;
  double elapsedTime;

  double simdTime;
  {
    const std::size_t Size = 1024 / 4;
    __m128 a[Size], b[Size], c[Size], x[Size], y[Size], z[Size];
    const __m128 initial = {1.f, 1.f, 1.f, 1.f};
    for (std::size_t i = 0; i != Size; ++i) {
      a[i] = b[i] = c[i] = x[i] = y[i] = z[i] = initial;
    }

    __m128 result;
    std::size_t count = 1;
    // Increase the size of the test until it runs for 0.1 seconds.
    do {
      count *= 2;
      result = _mm_setzero_ps();
      timer.tic();
      for (std::size_t n = 0; n != count; ++n) {
        for (std::size_t i = 0; i != Size; ++i) {
#if 0
          // This method is slower.
          t = _mm_mul_ps(a[i], x[i]);
          result = _mm_add_ps(result, t);
          t = _mm_mul_ps(b[i], y[i]);
          result = _mm_add_ps(result, t);
          t = _mm_mul_ps(c[i], z[i]);
          result = _mm_add_ps(result, t);
#endif
#if 0
          t = _mm_add_ps(_mm_add_ps(_mm_mul_ps(a[i], x[i]),
                                    _mm_mul_ps(b[i], y[i])),
                         _mm_mul_ps(c[i], z[i]));
          result = _mm_add_ps(result, t);
#else
          result += a[i] * x[i] + b[i] * y[i] + c[i] * z[i];
#endif
        }
      }
      elapsedTime = timer.toc();
    }
    while (elapsedTime < 0.1);

    simdTime = elapsedTime / (4 * Size * count) * 1e9;
    std::cout << "Meaningless result = "
              << *reinterpret_cast<const float*>(&result) << "\n"
              << "Time per SIMD dot product = " << simdTime
              << " nanoseconds.\n";
  }

  double scalarTime;
  {
    const std::size_t Size = 1024;
    std::array<float, 3> a[Size], b[Size];
    for (std::size_t i = 0; i != Size; ++i) {
      a[i] = b[i] = ext::filled_array<std::array<float, 3> >(1);
    }

    float result;
    std::size_t count = 1;
    // Increase the size of the test until it runs for 0.1 seconds.
    do {
      count *= 2;
      result = 0;
      timer.tic();
      for (std::size_t n = 0; n != count; ++n) {
        for (std::size_t i = 0; i != Size; ++i) {
          result += stlib::ext::dot(a[i], b[i]);
        }
      }
      elapsedTime = timer.toc();
    }
    while (elapsedTime < 0.1);

    scalarTime = elapsedTime / (Size * count) * 1e9;
    std::cout << "Meaningless result = "
              << *reinterpret_cast<const float*>(&result) << "\n"
              << "Time per scalar dot product = " << scalarTime
              << " nanoseconds.\n";
  }

  std::cout << "Speedup = " << scalarTime / simdTime << '\n';

  return 0;
}
