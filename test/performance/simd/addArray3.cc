// -*- C++ -*-

#include "stlib/ads/timer.h"
#include "stlib/ext/array.h"
#include "stlib/simd/operators.h"

#include <iostream>

#include <cstring>

USING_STLIB_EXT_ARRAY_MATH_OPERATORS;
using namespace stlib;

// Compare adding std::array<float, 3> and __m128. The speed is the same
// because the compiler is able to vectorize the operation for the former.
int
main()
{
  assert(sizeof(std::array<float, 3>) == sizeof(__m128));

  ads::Timer timer;
  double elapsedTime;

  double simdTime;
  {
    const std::size_t Size = 1024;
    std::array<float, 3> data[Size];
    // Note that it is important to zero to data to avoid NaNs.
    memset(&data[0], 0, sizeof(std::array<float, 3>) * Size);
    for (std::size_t i = 0; i != Size; ++i) {
      for (std::size_t j = 0; j != 3; ++j) {
        data[i][j] = 1;
      }
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
          result = _mm_add_ps(result,
                              *reinterpret_cast<const __m128*>(&data[i]));
#else
          result += _mm_load_ps(&data[i][0]);
#endif
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
    const std::size_t Size = 1024;
    std::array<float, 3> data[Size];
    for (std::size_t i = 0; i != Size; ++i) {
      for (std::size_t j = 0; j != 3; ++j) {
        data[i][j] = 1;
      }
    }

    std::array<float, 3> result;
    std::size_t count = 1;
    // Increase the size of the test until it runs for 0.1 seconds.
    do {
      count *= 2;
      result = ext::filled_array<std::array<float, 3> >(0);
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
              << result[0] << "\n"
              << "Time per scalar operation = " << scalarTime
              << " nanoseconds.\n";
  }

  std::cout << "Speedup = " << scalarTime / simdTime << '\n';

  return 0;
}
