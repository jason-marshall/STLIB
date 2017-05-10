// -*- C++ -*-

#include "stlib/numerical/specialFunctions/ExponentialForSmallArgument.h"
#include "stlib/ads/timer/Timer.h"

#include <iostream>
#include <fstream>

using namespace stlib;

int
main()
{
  double result = 0;
  ads::Timer timer;
  const int Count = 100000000;
  const double Epsilon = std::pow(std::numeric_limits<double>::epsilon(), 2);

  const double values[] = {0, 1e-16, 1e-15, 1e-14, 1e-13, 1e-12, 1e-11, 1e-10,
                           1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 5e-4,
                           1e-3, 5e-3, 1e-2, 5e-2, 1e-1, 5e-1, 1, 5, 10
                          };
  const int NumberOfValues = sizeof(values) / sizeof(double);

  std::cout
      << "CLOCKS_PER_SEC = " << CLOCKS_PER_SEC << "\n"
      << "Will call " << Count
      << " functions in each test.\n"
      << "Time given in nanoseconds.\n";

  numerical::ExponentialForSmallArgument<double> e;
  std::cout << "Theshholds:\n";
  e.printThreshholds(std::cout);
  std::cout << "Timings:\n";

  // Warm up.
  for (int n = 0; n != 1000; ++n) {
    result += e(0.0);
    result += std::exp(0.0);
  }

  double time1[NumberOfValues];
  for (int i = 0; i != NumberOfValues; ++i) {
    double x = values[i];
    const double negligible = x * Epsilon;
    timer.tic();
    for (int n = 0; n != Count; ++n) {
      result += e(x);
      // Do this to get a "different" argument each time.
      x += negligible;
    }
    time1[i] = timer.toc();
  }

  double time2[NumberOfValues];
  for (int i = 0; i != NumberOfValues; ++i) {
    double x = values[i];
    const double negligible = x * Epsilon;
    timer.tic();
    for (int n = 0; n != Count; ++n) {
      result += std::exp(x);
      // Do this to get a "different" argument each time.
      x += negligible;
    }
    time2[i] = timer.toc();
  }

  {
    std::ofstream out("ExponentialForSmallArgument.txt");
    for (int i = 0; i != NumberOfValues; ++i) {
      out << values[i] << " " << time1[i] / Count * 1e9 << "\n";
    }
  }
  {
    std::ofstream out("stdexp.txt");
    for (int i = 0; i != NumberOfValues; ++i) {
      out << values[i] << " " << time2[i] / Count * 1e9 << "\n";
    }
  }
  std::cout << "# Meaningless result = " << result << "\n";

  return 0;
}
