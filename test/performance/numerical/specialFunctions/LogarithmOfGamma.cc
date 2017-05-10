// -*- C++ -*-

#include "stlib/numerical/specialFunctions/Gamma.h"
#include "stlib/ads/timer/Timer.h"

#include <iostream>
#include <fstream>

using namespace stlib;

int
main()
{
  double result = 0;
  ads::Timer timer;
  const int Count = 10000000;
  const double Epsilon = std::pow(std::numeric_limits<double>::epsilon(), 2);

  const double values[] = {0, 1e-16, 1e-15, 1e-14, 1e-13,
                           1e-12, 1e-11, 1e-10, 1e-9,
                           1e-8, 1e-7, 1e-6, 1e-5,
                           1e-4, 1e-3, 1e-2, 1e-1,
                           1e0, 1e1, 1e2, 1e3, 1e4,
                           1e5, 1e6, 1e7, 1e8
                          };
  const int NumberOfValues = sizeof(values) / sizeof(double);

  std::cout
      << "CLOCKS_PER_SEC = " << CLOCKS_PER_SEC << "\n"
      << "Will call " << Count
      << " functions in each test.\n"
      << "Time given in nanoseconds.\n";

  numerical::LogarithmOfGamma<double> f;

  // Warm up.
  for (int n = 0; n != 1000; ++n) {
    result = f(1.0);
  }

  double time[NumberOfValues];
  for (int i = 0; i != NumberOfValues; ++i) {
    double x = values[i];
    const double negligible = x * Epsilon;
    timer.tic();
    for (int n = 0; n != Count; ++n) {
      result = f(x);
      // Do this to get a "different" argument each time.
      x += negligible;
    }
    time[i] = timer.toc();
  }

  {
    std::ofstream out("LogarithmOfGamma.txt");
    for (int i = 0; i != NumberOfValues; ++i) {
      out << values[i] << " " << time[i] / Count * 1e9 << "\n";
    }
  }

  std::cout << "# Meaningless result = " << result << "\n";

  return 0;
}
