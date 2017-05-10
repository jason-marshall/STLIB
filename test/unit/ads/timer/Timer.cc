// -*- C++ -*-

#include "stlib/ads/timer/Timer.h"

#include <iostream>

using namespace stlib;

int
main()
{
  std::cout << "sizeof(int) = " << sizeof(int) << '\n'
            << "sizeof(clock_t) = " << sizeof(clock_t) << '\n'
            << "CLOCKS_PER_SEC = " << CLOCKS_PER_SEC << '\n';

  {
    ads::Timer timer;
    timer.tic();
    double number = 1;
    for (std::size_t i = 0; i != 100; ++i) {
      number += number;
    }
    double time = timer.toc();
    std::cout << "Calculation took " << time << " seconds.\n";
  }

  {
    ads::Timer timer;

    timer.start();
    double number = 1;
    for (std::size_t i = 0; i != 100; ++i) {
      number += number;
    }
    timer.stop();
    std::cout << "First calculation took " << timer << " seconds.\n";

    timer.reset();
    timer.start();
    number = 1;
    for (std::size_t i = 0; i != 100; ++i) {
      number += number;
    }
    timer.stop();
    std::cout << "Second calculation took " << timer << " seconds.\n";
  }

  return 0;
}
