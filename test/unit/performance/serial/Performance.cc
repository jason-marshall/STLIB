// -*- C++ -*-

#define STLIB_PERFORMANCE
#include "stlib/performance/PerformanceSerial.h"

using namespace stlib::performance;

void
b()
{
  beginScope("b()");
  record("quantity", 3);
  endScope();
}

void
a()
{
  Scope scope("a()");
  {
    Event _("event1");
    record("quantity", 3);
  }
  start("event2");
  record("quantity", 3);
  stop();
  b();
}

int
main()
{
  start("main");
  record("quantity", 1);
  a();
  record("quantity", 1);
  stop(); // main

  print();
  printCsv(std::cout);

  return 0;
}
