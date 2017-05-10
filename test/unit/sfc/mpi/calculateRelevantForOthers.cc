// -*- C++ -*-

#include "stlib/sfc/gatherRelevant.h"

int
main(int argc, char* argv[])
{
  using stlib::sfc::calculateRelevantForOthers;
  
  MPI_Init(&argc, &argv);

  MPI_Comm const comm = MPI_COMM_WORLD;
  int const commSize = stlib::mpi::commSize();
  int const commRank = stlib::mpi::commRank();

  // Empty.
  {
    std::vector<std::size_t> const localCells;
    std::vector<std::size_t> const localRelevantCells;
    std::vector<std::size_t> relevantForOthers =
      calculateRelevantForOthers(0, localCells, localRelevantCells, comm);
    assert(relevantForOthers.empty());
    relevantForOthers =
      calculateRelevantForOthers(10, localCells, localRelevantCells, comm);
    assert(relevantForOthers.empty());
  }

  // One cell is relevant for all.
  {
    std::vector<std::size_t> localCells(10);
    for (std::size_t i = 0; i != localCells.size(); ++i) {
      localCells[i] = i;
    }
    std::vector<std::size_t> const localRelevantCells(1, 0);
    std::vector<std::size_t> relevantForOthers =
      calculateRelevantForOthers(localCells.size(), localCells,
                                 localRelevantCells, comm);
    if (commSize > 1) {
      assert(relevantForOthers == localRelevantCells);
    }
    else {
      assert(relevantForOthers.empty());
    }
  }

  // Each process has the rank. The rank is locally relevant.
  {
    std::vector<std::size_t> localCells(1, commRank);
    std::vector<std::size_t> const localRelevantCells(1, commRank);
    std::vector<std::size_t> relevantForOthers =
      calculateRelevantForOthers(commSize, localCells, localRelevantCells,
                                 comm);
    assert(relevantForOthers.empty());
  }
  
  // Each process has all ranks. The rank is locally relevant.
  {
    std::vector<std::size_t> localCells(commSize);
    for (std::size_t i = 0; i != localCells.size(); ++i) {
      localCells[i] = i;
    }
    std::vector<std::size_t> const localRelevantCells(1, commRank);
    std::vector<std::size_t> relevantForOthers =
      calculateRelevantForOthers(commSize, localCells, localRelevantCells,
                                 comm);
    std::vector<std::size_t> result;
    for (std::size_t i = 0; i != std::size_t(commSize); ++i) {
      if (i != std::size_t(commRank)) {
        result.push_back(i);
      }
    }
    assert(relevantForOthers == result);
  }
  
  MPI_Finalize();
  return 0;
}
