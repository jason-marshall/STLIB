// -*- C++ -*-

#include "stlib/sfc/gatherRelevant.h"

int
main(int argc, char* argv[])
{
  using stlib::sfc::calculateRelevantProcesses;
  
  MPI_Init(&argc, &argv);

  MPI_Comm const comm = MPI_COMM_WORLD;
  int const commSize = stlib::mpi::commSize();
  int const commRank = stlib::mpi::commRank();

  // Empty.
  {
    std::vector<std::size_t> const localCells;
    std::vector<std::size_t> const localRelevantCells;
    std::vector<int> relevantProcesses =
      calculateRelevantProcesses(0, localCells, localRelevantCells, comm);
    assert(relevantProcesses.empty());
    relevantProcesses =
      calculateRelevantProcesses(10, localCells, localRelevantCells, comm);
    assert(relevantProcesses.empty());
  }

  // Each process has the same cells. One cell is relevant for all.
  {
    std::vector<std::size_t> localCells(10);
    for (std::size_t i = 0; i != localCells.size(); ++i) {
      localCells[i] = i;
    }
    std::vector<std::size_t> const localRelevantCells(1, 0);
    std::vector<int> relevantProcesses =
      calculateRelevantProcesses(localCells.size(), localCells,
                                 localRelevantCells, comm);
    std::vector<int> result;
    for (int i = 1; i != commSize; ++i) {
      result.push_back((commRank + i) % commSize);
    }
    assert(relevantProcesses == result);
  }

  // Each process has the rank. Zero is relevant for all.
  {
    std::vector<std::size_t> localCells(1, commRank);
    std::vector<std::size_t> const localRelevantCells(1, 0);
    std::vector<int> relevantProcesses =
      calculateRelevantProcesses(commSize, localCells,
                                 localRelevantCells, comm);
    if (commRank == 0) {
      std::vector<int> result(commSize - 1);
      for (std::size_t i = 0; i != result.size(); ++i) {
        result[i] = i + 1;
      }
      assert(relevantProcesses == result);
    }
    else {
      assert(relevantProcesses.empty());
    }
  }

  // Each process has the rank. The rank is locally relevant.
  {
    std::vector<std::size_t> localCells(1, commRank);
    std::vector<std::size_t> const localRelevantCells(1, commRank);
    std::vector<int> relevantProcesses =
      calculateRelevantProcesses(commSize, localCells, localRelevantCells,
                                 comm);
    assert(relevantProcesses.empty());
  }
  
  // Each process has all ranks. The rank is locally relevant.
  {
    std::vector<std::size_t> localCells(commSize);
    for (std::size_t i = 0; i != localCells.size(); ++i) {
      localCells[i] = i;
    }
    std::vector<std::size_t> const localRelevantCells(1, commRank);
    std::vector<int> relevantProcesses =
      calculateRelevantProcesses(commSize, localCells, localRelevantCells,
                                 comm);
    std::vector<int> result;
    for (int i = 1; i != commSize; ++i) {
      result.push_back((commRank + i) % commSize);
    }
    assert(relevantProcesses == result);
  }
  
  MPI_Finalize();
  return 0;
}
