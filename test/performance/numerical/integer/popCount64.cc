// -*- C++ -*-

#include "stlib/numerical/integer/bits.h"

#include "stlib/ads/timer.h"

#include <iostream>

using namespace stlib;

int
main()
{

  ads::Timer timer;
  double elapsedTime;
  double time;

  const std::size_t Size = 1024;
  std::size_t data[Size];
  for (std::size_t i = 0; i != Size; ++i) {
    data[i] = i;
  }

  std::size_t result;
  std::size_t count = 1;
  // Increase the size of the test until it runs for 0.1 seconds.
  do {
    count *= 2;
    result = 0;
    timer.tic();
    for (std::size_t n = 0; n != count; ++n) {
      for (std::size_t i = 0; i != Size; ++i) {
        result += numerical::popCount(data[i]);
      }
    }
    elapsedTime = timer.toc();
  }
  while (elapsedTime < 0.1);

  time = elapsedTime / (Size * count) * 1e9;
  std::cout << "Meaningless result = " << result << "\n"
            << "Time per operation = " << time << " nanoseconds.\n";

  return 0;
}
