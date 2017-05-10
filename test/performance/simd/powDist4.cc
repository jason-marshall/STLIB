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
    __m128 data[Size];
    const __m128 initial = {1.f, 1.f, 1.f, 1.f};
    for (std::size_t i = 0; i != Size; ++i) {
      data[i] = initial;
    }
    __m128 r2 = {1, 0, 0, 0};
    __m128 center = {2, 2, 2, 2};

    __m128 result, t;
    std::size_t count = 1;
    // Increase the size of the test until it runs for 0.1 seconds.
    do {
      count *= 2;
      result = _mm_setzero_ps();
      timer.tic();
      for (std::size_t n = 0; n != count; ++n) {
        for (std::size_t i = 0; i != Size; ++i) {
          //t = _mm_sub_ps(center, data[i]);
          t = center - data[i];
          t = _mm_sub_ss(simd::dot(t, t), r2);
          result = _mm_add_ss(result, t);
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
    float data[Size];
    for (std::size_t i = 0; i != Size; ++i) {
      data[i] = 1;
    }
    float r2 = 1;
    float center[4] = {2, 2, 2, 2};

    float result;
    std::size_t count = 1;
    // Increase the size of the test until it runs for 0.1 seconds.
    do {
      count *= 2;
      result = 0;
      timer.tic();
      for (std::size_t n = 0; n != count; ++n) {
        for (std::size_t i = 0; i != Size; i += 4) {
          result += (center[0] - data[i]) * (center[0] - data[i]) +
                    (center[1] - data[i + 1]) * (center[1] - data[i + 1]) +
                    (center[2] - data[i + 2]) * (center[2] - data[i + 2]) +
                    (center[3] - data[i + 3]) * (center[3] - data[i + 3]) - r2;
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
