// -*- C++ -*-

#include "stlib/performance/SimpleTimer.h"

#include <iostream>

#include <cassert>

using namespace stlib::performance;

int
main()
{
  {
    SimpleTimer t;
    t.start();
    std::cout << "nanoseconds = " << std::flush;
    t.stop();
    std::cout << t.nanoseconds()
              << "\nelapsed = " << t.elapsed() << std::endl;
  }

  return 0;
}
