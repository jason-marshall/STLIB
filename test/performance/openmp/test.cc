// -*- C++ -*-

#include <iostream>

#ifdef _OPENMP
#include <omp.h>
#endif

int
main()
{
#ifdef _OPENMP
  std::cout << "Max threads = " << omp_get_max_threads() << '\n'
            << "Num processors = " << omp_get_num_procs() << '\n';
#else
  std::cout << "OpenMP is not available.\n";
#endif
  return 0;
}
