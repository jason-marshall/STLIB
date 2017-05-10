// -*- C++ -*-

#include "stlib/mpi/statistics.h"

#include <cassert>

using namespace stlib;

int
main(int argc, char* argv[])
{
  MPI_Init(&argc, &argv);

  int const commSize = mpi::commSize(MPI_COMM_WORLD);
  int const commRank = mpi::commRank(MPI_COMM_WORLD);

  if (commRank == 0) {
    mpi::printStatistics(std::cout, "zero", 0, MPI_COMM_NULL);
  }

  {
    float sum, mean, min, max;
    mpi::gatherStatistics(float(0), &sum, &mean, &min, &max, MPI_COMM_WORLD);
    if (commRank == 0) {
      assert(sum == 0);
      assert(mean == 0);
      assert(min == 0);
      assert(max == 0);
    }
  }
  {
    int sum, min, max;
    double mean;
    mpi::gatherStatistics(commRank, &sum, &mean, &min, &max, MPI_COMM_WORLD);
    if (commRank == 0) {
      assert(sum == commSize * (commSize - 1) / 2);
      assert(min == 0);
      assert(max == commSize - 1);
    }
    mpi::printStatistics(std::cout, "rank", commRank, MPI_COMM_WORLD);
  }

  // Empty vectors of values.
  {
    float sum, mean, min, max;
    mpi::gatherStatistics(std::vector<float>(), &sum, &mean, &min, &max,
                          MPI_COMM_WORLD);
    if (commRank == 0) {
      assert(sum == 0);
      assert(mean != mean);
      assert(min == std::numeric_limits<float>::max());
      assert(max == std::numeric_limits<float>::lowest());
    }
    mpi::printStatistics(std::cout, "MPI empty", std::vector<float>(),
                         MPI_COMM_WORLD);
    mpi::printStatistics(std::cout, "Serial empty", std::vector<float>(),
                         MPI_COMM_NULL);
  }
  // One non-empty vector of values.
  {
    std::vector<float> values;
    if (commRank == 0) {
      values.push_back(0);
    }
    float sum, mean, min, max;
    mpi::gatherStatistics(values, &sum, &mean, &min, &max, MPI_COMM_WORLD);
    if (commRank == 0) {
      assert(sum == 0);
      assert(mean == 0);
      assert(min == 0);
      assert(max == 0);
    }
  }
  // Single value, 0.
  {
    std::vector<float> values(1, float(0));
    float sum, mean, min, max;
    mpi::gatherStatistics(values, &sum, &mean, &min, &max, MPI_COMM_WORLD);
    if (commRank == 0) {
      assert(sum == 0);
      assert(mean == 0);
      assert(min == 0);
      assert(max == 0);
    }
  }
  // Single value, 1.
  {
    std::vector<float> values(1, float(1));
    float sum, mean, min, max;
    mpi::gatherStatistics(values, &sum, &mean, &min, &max, MPI_COMM_WORLD);
    if (commRank == 0) {
      assert(sum == commSize);
      assert(mean == 1);
      assert(min == 1);
      assert(max == 1);
    }
  }
  // Two values, 1.
  {
    std::vector<float> values(2, float(1));
    float sum, mean, min, max;
    mpi::gatherStatistics(values, &sum, &mean, &min, &max, MPI_COMM_WORLD);
    if (commRank == 0) {
      assert(sum == 2 * commSize);
      assert(mean == 1);
      assert(min == 1);
      assert(max == 1);
    }
    mpi::printStatistics(std::cout, "MPI ones", values, MPI_COMM_WORLD);
    mpi::printStatistics(std::cout, "Serial ones", values, MPI_COMM_NULL);
  }
  
  MPI_Finalize();
  return 0;
}
