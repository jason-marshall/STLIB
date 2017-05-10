// -*- C++ -*-

#include "stlib/mpi/partition.h"

#include <cassert>

using namespace stlib;

int
main(int argc, char* argv[])
{
  MPI_Init(&argc, &argv);

  int const commSize = mpi::commSize(MPI_COMM_WORLD);
  int const commRank = mpi::commRank(MPI_COMM_WORLD);

  // Empty vectors.
  {
    std::vector<std::size_t> objects;
    assert(mpi::partitionOrdered(&objects) == false);
    assert(objects.empty());
  }
  {
    std::vector<std::size_t> objects;
    assert(mpi::partitionExcess(&objects) == false);
    assert(objects.empty());
  }
  
  // Vectors of equal size.
  {
    std::vector<std::size_t> objects(1, commRank);
    assert(mpi::partitionOrdered(&objects) == false);
    assert(objects.size() == 1);
    assert(objects[0] == std::size_t(commRank));
  }
  {
    std::vector<std::size_t> objects(1, commRank);
    assert(mpi::partitionExcess(&objects) == false);
    assert(objects.size() == 1);
    assert(objects[0] == std::size_t(commRank));
  }
  
  // Vectors of unequal size.
  {
    std::vector<std::size_t> objects(commRank, commRank);
    std::size_t const size = mpi::allReduce(objects.size(), MPI_SUM);
    std::size_t const total = mpi::allReduce(stlib::ext::sum(objects), MPI_SUM);
    assert(mpi::partitionOrdered(&objects) == true);
    assert(size == mpi::allReduce(objects.size(), MPI_SUM));
    assert(total == mpi::allReduce(stlib::ext::sum(objects), MPI_SUM));
    std::size_t const minSize = mpi::reduce(objects.size(), MPI_MIN);
    std::size_t const maxSize = mpi::reduce(objects.size(), MPI_MAX);
    if (commRank == 0) {
      assert(maxSize - minSize <= 1);
    }
    
    std::vector<std::size_t> const gathered = mpi::gather(objects);
    if (commRank == 0) {
      std::size_t n = 0;
      for (std::size_t i = 0; i != std::size_t(commSize); ++i) {
        for (std::size_t j = 0; j != i; ++j, ++n) {
          assert(gathered[n] == i);
        }
      }
    }
  }
  {
    std::vector<std::size_t> objects(commRank, commRank);
    std::size_t const size = mpi::allReduce(objects.size(), MPI_SUM);
    std::size_t const total = mpi::allReduce(stlib::ext::sum(objects), MPI_SUM);
    assert(mpi::partitionExcess(&objects) == true);
    assert(size == mpi::allReduce(objects.size(), MPI_SUM));
    assert(total == mpi::allReduce(stlib::ext::sum(objects), MPI_SUM));
    std::size_t const minSize = mpi::reduce(objects.size(), MPI_MIN);
    std::size_t const maxSize = mpi::reduce(objects.size(), MPI_MAX);
    if (commRank == 0) {
      assert(maxSize - minSize <= 1);
    }
  }
  
  MPI_Finalize();
  return 0;
}
