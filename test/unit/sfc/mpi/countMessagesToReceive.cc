// -*- C++ -*-

#include "stlib/sfc/gatherRelevant.h"

int
main(int argc, char* argv[])
{
  using stlib::sfc::countMessagesToReceive;
  
  MPI_Init(&argc, &argv);

  MPI_Comm const comm = MPI_COMM_WORLD;
  int const commSize = stlib::mpi::commSize();
  int const commRank = stlib::mpi::commRank();

  // Empty.
  {
    std::vector<int> relevantProcesses;
    assert(countMessagesToReceive(relevantProcesses, comm) == 0);
  }

  // Next is relevant.
  {
    std::vector<int> relevantProcesses(1, (commRank + 1) % commSize);
    assert(countMessagesToReceive(relevantProcesses, comm) == 1);
  }

  // All others are relevant.
  {
    std::vector<int> relevantProcesses(commSize - 1);
    for (std::size_t i = 0; i != relevantProcesses.size(); ++i) {
      relevantProcesses[i] = (commRank + i + 1) % commSize;
    }
    assert(countMessagesToReceive(relevantProcesses, comm) ==
           std::size_t(commSize - 1));
  }
  
  MPI_Finalize();
  return 0;
}
