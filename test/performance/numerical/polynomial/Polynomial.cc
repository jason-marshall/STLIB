// -*- C++ -*-

#include "stlib/numerical/polynomial/Polynomial.h"
#include "stlib/numerical/polynomial/PolynomialGenericOrder.h"
#include "stlib/ads/timer/Timer.h"

#include <iostream>

using namespace stlib;

template<std::size_t _Order>
void
timeStatic()
{
  std::cout << "Order = " << _Order << '\n';
  std::array < double, _Order + 1 > c;
  std::fill(c.begin(), c.end(), 1);
  numerical::Polynomial<_Order> p(c);
  // Warm up.
  double result = 0;
  for (double x = 0; x != 100; ++x) {
    result += p(x);
  }
  // Time.
  ads::Timer timer;
  timer.tic();
  for (double x = 0; x != 1000; ++x) {
    result += p(x);
  }
  timer.toc();
  std::cout << "Time per evaluation = " << timer * 1e9 / 1000
            << " nanoseconds.\n"
            << "Meaningless result = " << result << "\n\n";
}

void
timeGeneric(const std::size_t order)
{
  std::cout << "Order = " << order << '\n';
  std::vector<double> c(order + 1);
  std::fill(c.begin(), c.end(), 1);
  numerical::PolynomialGenericOrder<> p(c);
  // Warm up.
  double result = 0;
  for (double x = 0; x != 100; ++x) {
    result += p(x);
  }
  // Time.
  ads::Timer timer;
  timer.tic();
  for (double x = 0; x != 1000; ++x) {
    result += p(x);
  }
  timer.toc();
  std::cout << "Time per evaluation = " << timer * 1e9 / 1000
            << " nanoseconds.\n"
            << "Meaningless result = " << result << "\n\n";
}

int
main()
{
  std::cout << "------------------------------------------------------------\n"
            << "Static order:\n";
  timeStatic<0>();
  timeStatic<1>();
  timeStatic<2>();
  timeStatic<3>();
  std::cout << "------------------------------------------------------------\n"
            << "Generic order:\n";
  timeGeneric(0);
  timeGeneric(1);
  timeGeneric(2);
  timeGeneric(3);
  return 0;
}
