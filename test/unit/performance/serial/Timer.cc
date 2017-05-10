// -*- C++ -*-

#include "stlib/performance/Timer.h"

#include <iostream>

#include <cassert>

using namespace stlib::performance;

int
main()
{
  {
    Timer t;
    assert(! t.isStopped());
    t.stop();
    assert(t.isStopped());
  }
  {
    Timer t;
    std::cout << "construct, nanoseconds() = " << std::flush;
    std::cout << t.nanoseconds() << std::endl;
  }
  {
    Timer t;
    std::cout << "construct, elapsed() = " << std::flush;
    std::cout << t.elapsed() << std::endl;
  }
  {
    Timer t;
    std::cout << "construct, start(), elapsed() = " << std::flush;
    t.start();
    std::cout << t.elapsed() << std::endl;
  }
  {
    Timer t;
    std::cout << "construct, stop(), elapsed() = " << std::flush;
    t.stop();
    std::cout << t.elapsed() << std::endl;
  }
  {
    Timer t;
    std::cout << "construct, stop(), resume(), elapsed() = " << std::flush;
    t.stop();
    t.resume();
    std::cout << t.elapsed() << std::endl;
  }

  return 0;
}
