// -*- C++ -*-

#include "stlib/simd/functions.h"
#include "stlib/ads/timer.h"

#include <iostream>


using namespace stlib;

int
main()
{

  ads::Timer timer;
  double elapsedTime;

  double simdTime;
  {
    const std::size_t Size = 1024;
    __m128 a[Size], b[Size];
    const __m128 one = {1.f, 1.f, 1.f, 1.f};
    for (std::size_t i = 0; i != Size; ++i) {
      a[i] = one;
      b[i] = one;
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
          result = _mm_add_ss(result, simd::dot(a[i], b[i]));
        }
      }
      elapsedTime = timer.toc();
    }
    while (elapsedTime < 0.1);

    simdTime = elapsedTime / (Size * count) * 1e9;
    std::cout << "Meaningless result = "
              << *reinterpret_cast<const float*>(&result) << "\n"
              << "Time per SIMD operation = " << simdTime
              << " nanoseconds.\n";
  }

  double scalarTime;
  {
    const std::size_t Size = 4 * 1024;
    float a[Size], b[Size];
    for (std::size_t i = 0; i != Size; ++i) {
      a[i] = 1;
      b[i] = 1;
    }

    float result;
    std::size_t count = 1;
    // Increase the size of the test until it runs for 0.1 seconds.
    do {
      count *= 2;
      result = 0;
      timer.tic();
      for (std::size_t n = 0; n != count; ++n) {
        for (std::size_t i = 0; i != Size; i += 4) {
          result += a[i] * b[i] + a[i + 1] * b[i + 1] +
                    a[i + 2] * b[i + 2] + a[i + 3] * b[i + 3];
        }
      }
      elapsedTime = timer.toc();
    }
    while (elapsedTime < 0.1);

    scalarTime = elapsedTime / (Size / 4 * count) * 1e9;
    std::cout << "Meaningless result = "
              << *reinterpret_cast<const float*>(&result) << "\n"
              << "Time per scalar operation = " << scalarTime
              << " nanoseconds.\n";
  }

  std::cout << "Speedup = " << scalarTime / simdTime << '\n';

  return 0;
}
