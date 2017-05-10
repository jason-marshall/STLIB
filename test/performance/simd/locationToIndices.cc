// -*- C++ -*-

#include "stlib/ext/array.h"
#include "stlib/ads/timer.h"
#include "stlib/simd/operators.h"

#include <iostream>

using namespace stlib;

int
main()
{
  const std::size_t D = 3;
  typedef std::array<float, D> Point;

  ads::Timer timer;
  double elapsedTime;

  const std::size_t Size = 1024;
  Point data[Size];
  for (std::size_t i = 0; i != Size; ++i) {
    data[i] = ext::filled_array<Point>(i);
  }

  Point lowerCorner, inverseCellLengths;
  for (std::size_t i = 0; i != D; ++i) {
    lowerCorner[i] = float(rand()) / RAND_MAX;
    inverseCellLengths[i] = 1 + 0.001 * float(rand()) / RAND_MAX;
  }

  double simdTime;
  {
    std::size_t result;
    std::size_t count = 1;
    __m128 zero = _mm_set1_ps(0);
    __m128 upper = _mm_set1_ps(Size);
    __m128 lc = _mm_set_ps(0, lowerCorner[2], lowerCorner[1], lowerCorner[0]);
    __m128 icl = _mm_set_ps(0, inverseCellLengths[2], inverseCellLengths[1],
                            inverseCellLengths[0]);
    __m128 d, index;
    const float* idx = reinterpret_cast<const float*>(&index);
    // Increase the size of the test until it runs for 0.1 seconds.
    do {
      count *= 2;
      result = 0;
      timer.tic();
      for (std::size_t n = 0; n != count; ++n) {
        for (std::size_t m = 0; m != Size; ++m) {
#if 0
          // Very slow.
          d = _mm_load_ps(&data[m][0]);
          reinterpret_cast<float*>(&d)[3] = 0;
#else
          // Not too bad.
          d = _mm_set_ps(0, data[m][2], data[m][1], data[m][0]);
#endif

#if 1
          index = _mm_min_ps(upper,
                             _mm_max_ps(zero, d - lc) * icl);
#else
          index = d;
#endif

          // Casting to an int is a little less expensive.
          result += int(idx[0]) + int(idx[1]) + int(idx[2]);
        }
      }
      elapsedTime = timer.toc();
    }
    while (elapsedTime < 0.1);

    simdTime = elapsedTime / (Size * count) * 1e9;
    std::cout << "Meaningless result = " << result << "\n"
              << "Time per SIMD operation = " << simdTime
              << " nanoseconds.\n";
  }

  double scalarTime;
  {
    std::size_t result;
    std::size_t count = 1;
    // Increase the size of the test until it runs for 0.1 seconds.
    do {
      count *= 2;
      result = 0;
      timer.tic();
      for (std::size_t n = 0; n != count; ++n) {
        for (std::size_t m = 0; m != Size; ++m) {
          for (std::size_t i = 0; i != D; ++i) {
            result += std::min(Size,
                               std::size_t(std::max(0.f,
                                                    data[m][i] -
                                                    lowerCorner[i]) *
                                           inverseCellLengths[i]));
          }
        }
      }
      elapsedTime = timer.toc();
    }
    while (elapsedTime < 0.1);

    scalarTime = elapsedTime / (Size * count) * 1e9;
    std::cout << "Meaningless result = " << result << "\n"
              << "Time per scalar operation = " << scalarTime
              << " nanoseconds.\n";
  }

  std::cout << "Speedup = " << scalarTime / simdTime << '\n';

  return 0;
}
