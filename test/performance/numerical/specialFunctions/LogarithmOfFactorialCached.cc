// -*- C++ -*-

#include "stlib/numerical/specialFunctions/LogarithmOfFactorialCached.h"
#include "stlib/ads/timer/Timer.h"

#include <algorithm>
#include <iostream>
#include <vector>
#include <fstream>

using namespace stlib;

int
main()
{
  double result = 0;
  ads::Timer timer;
  const int Count = 1000000;
  const int Size = 1000;

  numerical::LogarithmOfFactorialCached<double> f(Size);

  std::vector<int> arguments(Size);
  for (std::vector<int>::size_type i = 0; i != arguments.size(); ++i) {
    arguments[i] = i;
  }
  std::random_shuffle(arguments.begin(), arguments.end());

  // Warm up.
  for (int n = 0; n != 100; ++n) {
    for (int i = 0; i != Size; ++i) {
      result += f(arguments[i]);
    }
  }

  timer.tic();
  for (int n = 0; n != Count; ++n) {
    for (int i = 0; i != Size; ++i) {
      result += f(arguments[i]);
    }
  }
  const double elapsedTime = timer.toc();


  std::cout << "Meaningless result = " << result << "\n"
            << "CLOCKS_PER_SEC = " << CLOCKS_PER_SEC << "\n"
            << "Made " << Count* Size << " function calls.\n"
            << "Time per function call = "
            << elapsedTime / (Count * Size) * 1e9 << " nanoseconds.\n";

  return 0;
}
