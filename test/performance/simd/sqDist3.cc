// -*- C++ -*-

#include "stlib/simd/functions.h"
#include "stlib/ext/array.h"
#include "stlib/ads/timer/Timer.h"

#include <vector>

#include <cstdlib>

using namespace stlib;

int
main()
{
  typedef float T;
  const std::size_t Size = 1024;

  // Make two arrays of random points.
  __m128 a[Size], b[Size];
  for (std::size_t i = 0; i != Size; ++i) {
    a[i] = _mm_set_ps(0, rand(), rand(), rand());
    b[i] = _mm_set_ps(0, rand(), rand(), rand());
  }

  ads::Timer timer;
  std::size_t count = 1;
  T result = 0;
  double elapsedTime;
  __m128 t;
  do {
    count *= 2;
    timer.tic();
    for (std::size_t i = 0; i != count; ++i) {
      for (std::size_t j = 0; j != Size; ++j) {
        t = a[j] - b[j];
        t = simd::dot(t, t);
        result += *reinterpret_cast<const float*>(&t);
      }
    }
    elapsedTime = timer.toc();
  }
  while (elapsedTime < 0.1);

  std::cout
      << "Meaningless result = " << result << "\n"
      << "CLOCKS_PER_SEC = " << CLOCKS_PER_SEC << "\n"
      << count* Size << " operations in "
      << elapsedTime << " seconds.\n"
      << "Time per operation in nanoseconds:\n";
  std::cout.setf(std::ios_base::fixed, std::ios_base::floatfield);
  std::cout.precision(1);
  std::cout << elapsedTime / (count * Size) * 1e9 << "\n";

  return 0;
}
