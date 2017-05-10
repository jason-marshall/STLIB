// -*- C++ -*-

#include "stlib/ext/array.h"
#include "stlib/ads/timer/Timer.h"

#include <vector>

#include <cstring>

using namespace stlib;

int
main()
{
  typedef std::array<T, D> Point;
  const std::size_t Size = 1024;

  // Make two vectors of random points.
  std::vector<Point> a(Size), b(Size);
  // First invalidate the memory.
  memset(&a[0], 0xFFFF, a.size() * sizeof(Point));
  memset(&b[0], 0xFFFF, b.size() * sizeof(Point));
  for (std::size_t i = 0; i != a.size(); ++i) {
    for (std::size_t j = 0; j != D; ++j) {
      a[i][j] = rand();
      b[i][j] = rand();
    }
  }

  ads::Timer timer;
  std::size_t count = 1;
  T result = 0;
  double elapsedTime;
  do {
    count *= 2;
    timer.tic();
    for (std::size_t i = 0; i != count; ++i) {
      for (std::size_t j = 0; j != a.size(); ++j) {
        result += stlib::ext::euclideanDistance(a[j], b[j]);
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
