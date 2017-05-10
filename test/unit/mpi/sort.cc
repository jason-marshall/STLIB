// -*- C++ -*-

#include "stlib/mpi/sort.h"

#include <cassert>

using namespace stlib;

int
main(int argc, char* argv[])
{
  MPI_Init(&argc, &argv);

  int const commSize = mpi::commSize(MPI_COMM_WORLD);
  int const commRank = mpi::commRank(MPI_COMM_WORLD);

  {
    std::vector<std::size_t> input(1, commRank);
    std::vector<std::size_t> output;
    mpi::mergeSortedSequentialScan(input, &output);
    if (commRank == 0) {
      assert(output.size() == std::size_t(commSize));
      for (std::size_t i = 0; i != output.size(); ++i) {
        assert(output[i] == i);
      }
    }
    else {
      assert(output.size() == 0);
      assert(output.capacity() == 0);
    }
  }
  {
    std::vector<std::size_t> input(1, commRank);
    std::vector<std::size_t> output;
    mpi::mergeSortedSequentialScan(input, &output);
    if (commRank == 0) {
      assert(output.size() == std::size_t(commSize));
      for (std::size_t i = 0; i != output.size(); ++i) {
        assert(output[i] == i);
      }
    }
    else {
      assert(output.size() == 0);
      assert(output.capacity() == 0);
    }
  }

  // _mergeSortedPairs()
  // Note: The same test is run on each process.
  {
    using stlib::mpi::_mergeSortedPairs;
    typedef std::pair<std::size_t, std::size_t> Pair;
    {
      std::vector<Pair> a;
      std::vector<Pair> b;
      std::vector<Pair> output;
      _mergeSortedPairs(a, b, &output);
      assert(output.empty());
    }
    {
      std::vector<Pair> a = {{1, 1}};
      std::vector<Pair> b;
      std::vector<Pair> output;
      _mergeSortedPairs(a, b, &output);
      assert(output == a);
      _mergeSortedPairs(b, a, &output);
      assert(output == a);
    }
    {
      std::vector<Pair> a = {{1, 1}};
      std::vector<Pair> b = {{1, 1}};
      std::vector<Pair> output;
      _mergeSortedPairs(a, b, &output);
      assert(output == (std::vector<Pair>{{1, 2}}));
    }
    {
      std::vector<Pair> a = {{1, 1}, {2, 2}, {3, 3}, {4, 4}};
      std::vector<Pair> b = {{2, 1}, {3, 1}, {5, 1}};
      std::vector<Pair> const result = {{1, 1}, {2, 3}, {3, 4}, {4, 4}, {5, 1}};
      std::vector<Pair> output;
      _mergeSortedPairs(a, b, &output);
      assert(output == result);
      _mergeSortedPairs(b, a, &output);
      assert(output == result);
    }
  }

  // mergeSorted()
  {
    using stlib::mpi::mergeSorted;
    typedef std::pair<std::size_t, std::size_t> Pair;
    {
      std::vector<Pair> input;
      std::vector<Pair> output;
      mergeSorted(input, &output);
      assert(output.empty());
    }
    {
      std::vector<Pair> input = {{1, 1}};
      std::vector<Pair> output;
      mergeSorted(input, &output);
      if (commRank == 0) {
        assert(output == (std::vector<Pair>{{1, commSize}}));
      }
      else {
        assert(output.empty());
      }
    }
    {
      std::vector<Pair> input = {{commRank, commRank}};
      std::vector<Pair> output;
      mergeSorted(input, &output);
      if (commRank == 0) {
        assert(output.size() == std::size_t(commSize));
        for (std::size_t i = 0; i != output.size(); ++i) {
          assert(output[i] == (Pair{i, i}));
        }
      }
      else {
        assert(output.empty());
      }
    }
  }

  MPI_Finalize();
  return 0;
}
