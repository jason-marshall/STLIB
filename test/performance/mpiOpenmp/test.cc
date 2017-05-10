// -*- C++ -*-

#include <iostream>

#include <mpi.h>
#ifdef _OPENMP
#include <omp.h>
#endif

int
main(int argc, char* argv[])
{
  MPI_Init(&argc, &argv);

  int size = 0;
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  if (rank == 0) {
    std::cout << "Num MPI processes = " << size << '\n';
#ifdef _OPENMP
    std::cout << "OpenMP max threads = " << omp_get_max_threads() << '\n'
              << "OpenMP num processors = " << omp_get_num_procs() << '\n';
#else
    std::cout << "OpenMP is not available.\n";
#endif
  }

  MPI_Finalize();
  return 0;
}
