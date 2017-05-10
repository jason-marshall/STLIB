// -*- C++ -*-

#include <iostream>
#include <limits>
#include <vector>

#include "stlib/ads/timer.h"
#include "stlib/ads/utility/ParseOptionsArguments.h"

using namespace stlib;

int
main(int argc, char* argv[])
{
  ads::ParseOptionsArguments parser(argc, argv);

  std::size_t count = 0;
  parser.getArgument(&count);
  assert(count != 0);

  std::vector<double> x(count, 0);
  for (std::size_t i = 0; i < x.size(); i += 2) {
    x[i] = std::numeric_limits<double>::quiet_NaN();
  }

  ads::Timer timer;
  std::size_t result = 0;
  timer.tic();
  for (std::size_t i = 0; i != x.size(); ++i) {
    result += x[i] != x[i];
  }
  double elapsedTime = timer.toc();
  assert(result == x.size() / 2);

  std::cout << "Value = " << std::numeric_limits<double>::quiet_NaN() << '\n'
            << "Meaningless result = " << result << '\n'
            << "Time per comparison = " << elapsedTime / x.size() * 1e9
            << " nanoseconds.\n";

  return 0;
}

