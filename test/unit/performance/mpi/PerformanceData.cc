// -*- C++ -*-

#include "stlib/performance/PerformanceDataMpi.h"

using namespace stlib;

int
main(int argc, char* argv[])
{
  MPI_Init(&argc, &argv);
  
  performance::PerformanceData p;
  assert(p.empty());
  
  p.start("main");
  p.record("2", 1);
  p.record("2", 1);
  p.stop(); // main

  print(std::cout, p, MPI_COMM_WORLD);
  printCsv(std::cout, p, MPI_COMM_WORLD);

  MPI_Finalize();
  return 0;
}
