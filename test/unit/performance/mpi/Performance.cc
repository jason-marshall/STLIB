// -*- C++ -*-

#define STLIB_PERFORMANCE
#include "stlib/performance/PerformanceMpi.h"

using namespace stlib;

int
main(int argc, char* argv[])
{
  MPI_Init(&argc, &argv);

  performance::start("main");
  performance::record("quantity", 1);

  performance::beginScope("function");
  performance::start("event1");
  performance::record("quantity", 3);
  performance::stop();
  performance::start("event2");
  performance::record("quantity", 3);
  performance::stop();
  performance::endScope();

  performance::record("quantity", 1);

  performance::stop(); // main

  performance::print();
  performance::printCsv();

  MPI_Finalize();
  return 0;
}
