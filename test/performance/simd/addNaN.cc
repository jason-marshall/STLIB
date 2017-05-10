// -*- C++ -*-

#include "stlib/ads/timer.h"
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
    const std::size_t Size = 1024;
    __m128 data[Size];
    for (std::size_t i = 0; i != Size; ++i) {
      __m128 a = _mm_set1_ps(std::numeric_limits<float>::quiet_NaN());
      data[i] = a;
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
          //result = _mm_add_ps(result, data[i]);
          result += data[i];
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
      data[i] = std::numeric_limits<float>::quiet_NaN();
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
          result += data[i];
        }
      }
      elapsedTime = timer.toc();
    }
    while (elapsedTime < 0.1);

    scalarTime = elapsedTime / (Size * count) * 1e9;
    std::cout << "Meaningless result = "
              << *reinterpret_cast<const float*>(&result) << "\n"
              << "Time per scalar operation = " << scalarTime
              << " nanoseconds.\n";
  }

  std::cout << "Speedup = " << 4 * scalarTime / simdTime << '\n';

  return 0;
}
