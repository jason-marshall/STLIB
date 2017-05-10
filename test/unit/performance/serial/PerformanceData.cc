// -*- C++ -*-

#include "stlib/performance/PerformanceDataSerial.h"

#include <cassert>

using namespace stlib;

int
main()
{
  performance::PerformanceData p;
  assert(p.empty());
  
  p.start("main");
  p.record("2", 1);
  p.record("2", 1);
  p.stop(); // main
  try {
    p.stop(); // erroneous
    assert(false);
  }
  catch (std::runtime_error) {
  }

  print(std::cout, p);
  printCsv(std::cout, p);

  return 0;
}
