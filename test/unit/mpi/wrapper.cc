// -*- C++ -*-

#include "stlib/mpi/wrapper.h"

#include <cassert>


using namespace stlib;

template<typename _T>
struct Value {
  _T value;
};


int
main(int argc, char* argv[])
{
  MPI_Init(&argc, &argv);

  int const commSize = mpi::commSize();
  int const commRank = mpi::commRank();

  // Barrier.
  mpi::barrier();

  // Send/receive.
  if (commSize >= 2) {
    // Use the interface that checks the size.
    {
      int const sent = 7;
      std::size_t const count = 1;
      MPI_Datatype const datatype = mpi::Data<int>::type();

      if (commRank == 0) {
        mpi::send(&sent, count, datatype, 1, 0);
      }
      else if (commRank == 1) {
        int received = 0;
        mpi::recv(&received, count, datatype, 0, 0);
      }
    }
    // Use the std::vector interface that deduces the datatype.
    {
      std::vector<int> input;
      input.push_back(2);
      input.push_back(3);
      input.push_back(5);

      if (commRank == 0) {
        mpi::send(input, 1, 0);
      }
      else if (commRank == 1) {
        std::vector<int> const output = mpi::recv<int>(0, 0);
        assert(output == input);
      }
    }
  }

  // Verify that we get an overflow error when sending or receiving a buffer
  // that is too large.
  {
    try {
      mpi::send(0, std::size_t(std::numeric_limits<int>::max()) + 1, MPI_BYTE,
                commRank, 0);
      throw std::runtime_error("Expected an overflow error.");
    }
    catch (std::overflow_error) {
    }
    try {
      mpi::recv(0, std::size_t(std::numeric_limits<int>::max()) + 1, MPI_BYTE,
                commRank, 0);
      throw std::runtime_error("Expected an overflow error.");
    }
    catch (std::overflow_error) {
    }
  }

  // Sendrecv.
  if (commSize >= 2) {
    // Use the interface that checks the size.
    {
      std::size_t const count = 1;
      MPI_Datatype const datatype = mpi::Data<int>::type();

      int recv = -1;
      if (commRank == 0) {
        mpi::sendRecv(&commRank, count, datatype, 1, 0,
                      &recv, count, datatype, 1, 0);
        assert(recv == 1);
      }
      else if (commRank == 1) {
        mpi::sendRecv(&commRank, count, datatype, 0, 0,
                      &recv, count, datatype, 0, 0);
        assert(recv == 0);
      }
    }
    // Send a single object.
    {
      MPI_Datatype const datatype = mpi::Data<int>::type();
      int recv = -1;
      if (commRank == 0) {
        mpi::sendRecv(commRank, datatype, 1, 0,
                      &recv, datatype, 1, 0);
        assert(recv == 1);
      }
      else if (commRank == 1) {
        mpi::sendRecv(commRank, datatype, 0, 0,
                      &recv, datatype, 0, 0);
        assert(recv == 0);
      }
    }
    // Send a single object. Deduce the data type.
    {
      int recv = -1;
      if (commRank == 0) {
        mpi::sendRecv(commRank, 1, 0, &recv, 1, 0);
        assert(recv == 1);
      }
      else if (commRank == 1) {
        mpi::sendRecv(commRank, 0, 0, &recv, 0, 0);
        assert(recv == 0);
      }
    }
    // Send and receive ranks are the same.
    {
      int recv = -1;
      if (commRank == 0) {
        mpi::sendRecv(commRank, &recv, 1, 0);
        assert(recv == 1);
      }
      else if (commRank == 1) {
        mpi::sendRecv(commRank, &recv, 0, 0);
        assert(recv == 0);
      }
    }
    // Send and receive vectors.
    {
      std::vector<int> send(2, commRank);
      std::vector<int> recv(2);
      if (commRank == 0) {
        mpi::sendRecv(send, 1, &recv, 1, 0);
        assert(recv.size() == 2);
        assert(recv[0] == 1);
        assert(recv[1] == 1);
      }
      else if (commRank == 1) {
        mpi::sendRecv(send, 0, &recv, 0, 0);
        assert(recv.size() == 2);
        assert(recv[0] == 0);
        assert(recv[1] == 0);
      }
    }
  }

  // Probe.
  if (commSize >= 2) {
    std::vector<int> const input = {2, 3, 5};
    if (commRank == 0) {
      mpi::send(input, 1, 0);
    }
    else if (commRank == 1) {
      MPI_Status status = mpi::probe(0, 0);
      assert(status.MPI_SOURCE == 0);
      assert(status.MPI_TAG == 0);
      assert(std::size_t(mpi::getCount<int>(status)) == input.size());
      std::vector<int> const output = mpi::recv<int>(0, 0);
      assert(output == input);
    }
    MPI_Barrier(MPI_COMM_WORLD);
  }

  // Non-blocking send and receive.
  if (commSize >= 2) {
    // Use the std::vector interface that deduces the datatype.
    {
      std::vector<int> input;
      input.push_back(2);
      input.push_back(3);
      input.push_back(5);

      // Check the counts.
      if (commRank == 0) {
#if 0
        // CONTINUE: Checking the send count does not work on Darwin.
        MPI_Request request = mpi::iSend(input, 1, 0);
        mpi::wait<int>(&request, input.size());
#else
        mpi::iSend(input, 1, 0);
#endif
      }
      else if (commRank == 1) {
        std::vector<int> output(input.size());
        MPI_Request request = mpi::iRecv(&output, 0, 0);
        mpi::wait<int>(&request, input.size());
        assert(output == input);
      }
      MPI_Barrier(MPI_COMM_WORLD);

      // Don't check the counts.
      if (commRank == 0) {
        MPI_Request request = mpi::iSend(input, 1, 0);
        mpi::wait(&request);
      }
      else if (commRank == 1) {
        std::vector<int> output(input.size());
        MPI_Request request = mpi::iRecv(&output, 0, 0);
        mpi::wait(&request);
        assert(output == input);
      }
      MPI_Barrier(MPI_COMM_WORLD);

      // Don't check the counts. Use waitAll().
      if (commRank == 0) {
        std::vector<MPI_Request> requests = {mpi::iSend(input, 1, 0)};
        mpi::waitAll(&requests);
      }
      else if (commRank == 1) {
        std::vector<int> output(input.size());
        std::vector<MPI_Request> requests = {mpi::iRecv(&output, 0, 0)};
        mpi::waitAll(&requests);
        assert(output == input);
      }
      MPI_Barrier(MPI_COMM_WORLD);
    }
  }

  // Gather with a built-in type.
  {
    std::size_t const rank = commRank;
    std::vector<std::size_t> const ranks = mpi::gather(rank);
    if (rank == 0) {
      assert(ranks.size() == std::size_t(commSize));
      for (std::size_t i = 0; i != ranks.size(); ++i) {
        assert(ranks[i] == i);
      }
    }
    else {
      assert(ranks.empty());
    }
  }
  
  // Gather with a specialized type.
  {
    typedef std::array<std::size_t, 1> T;
    T const rank = {{std::size_t(commRank)}};
    std::vector<T> const ranks = mpi::gather(rank);
    if (rank[0] == 0) {
      assert(ranks.size() == std::size_t(commSize));
      for (std::size_t i = 0; i != ranks.size(); ++i) {
        assert(ranks[i][0] == i);
      }
    }
    else {
      assert(ranks.empty());
    }
  }
  
  // Gather with a POD type.
  {
    typedef Value<std::size_t> T;
    T const rank = {std::size_t(commRank)};
    std::vector<T> const ranks = mpi::gather(rank);
    if (rank.value == 0) {
      assert(ranks.size() == std::size_t(commSize));
      for (std::size_t i = 0; i != ranks.size(); ++i) {
        assert(ranks[i].value == i);
      }
    }
    else {
      assert(ranks.empty());
    }
  }

  // Wrapper for MPI_Gatherv().
  {
    std::vector<std::size_t> send(10, commRank);
    std::vector<std::size_t> const receive = mpi::gather(send);
    if (commRank == 0) {
      assert(receive.size() == send.size() * commSize);
      std::size_t n = 0;
      for (std::size_t i = 0; i != std::size_t(commSize); ++i) {
        for (std::size_t j = 0; j != send.size(); ++j, ++n) {
          assert(receive[n] == i);
        }
      }
    }
    else {
      assert(receive.empty());
    }
  }
  
  // allGather() with single object.
  {
    std::size_t const rank = commRank;
    std::vector<std::size_t> const ranks = mpi::allGather(rank);
    assert(ranks.size() == std::size_t(commSize));
    for (std::size_t i = 0; i != ranks.size(); ++i) {
      assert(ranks[i] == i);
    }
  }

  // allGather() with vector.
  {
    std::vector<std::size_t> rank(1, commRank);
    std::vector<std::size_t> const ranks = mpi::allGather(rank);
    assert(ranks.size() == std::size_t(commSize));
    for (std::size_t i = 0; i != ranks.size(); ++i) {
      assert(ranks[i] == i);
    }
  }
  {
    std::vector<std::size_t> rank(commRank, commRank);
    std::vector<std::size_t> const ranks = mpi::allGather(rank);
    assert(ranks.size() == std::size_t(commSize * (commSize - 1) / 2));
    std::size_t n = 0;
    for (std::size_t i = 0; i != std::size_t(commSize); ++i) {
      for (std::size_t j = 0; j != i; ++j) {
        assert(ranks[n++] == i);
      }
    }
    assert(n == ranks.size());
  }
  
  // allGather() with PackedArrayOfArrays.
  {
    std::vector<std::size_t> rank(1, commRank);
    container::PackedArrayOfArrays<std::size_t> const ranks =
      mpi::allGatherPacked(rank);
    assert(ranks.size() == std::size_t(commSize));
    for (std::size_t i = 0; i != ranks.size(); ++i) {
      assert(ranks.size(i) == 1);
      assert(ranks(i, 0) == i);
    }
  }
  {
    std::vector<std::size_t> rank(commRank, commRank);
    container::PackedArrayOfArrays<std::size_t> const ranks =
      mpi::allGatherPacked(rank);
    assert(ranks.size() == std::size_t(commSize * (commSize - 1) / 2));
    for (std::size_t i = 0; i != std::size_t(commSize); ++i) {
      assert(ranks.size(i) == i);
      for (std::size_t j = 0; j != i; ++j) {
        assert(ranks(i, j) == i);
      }
    }
  }
  
  // Simple wrapper for MPI_Allgather().
  {
    std::size_t const rank = commRank;
    std::vector<std::size_t> ranks(commSize);
    mpi::allGather(&rank, 1, mpi::Data<std::size_t>::type(), &ranks[0], 1,
                   mpi::Data<std::size_t>::type());
    for (std::size_t i = 0; i != ranks.size(); ++i) {
      assert(ranks[i] == i);
    }
  }
  
  // Reduce with a built-in type.
  {
    std::size_t const sum = commSize * (commSize - 1) / 2;
    // root == 0.
    {
      std::size_t const total = mpi::reduce(commRank, MPI_SUM);
      if (commRank == 0) {
        assert(total == sum);
      }
    }
    // root == 1.
    {
      int const root = 1 % commSize;
      std::size_t const total = mpi::reduce(commRank, MPI_SUM, MPI_COMM_WORLD,
                                            root);
      if (commRank == root) {
        assert(total == sum);
      }
    }
  }
  
  // Reduce with a vector of values.
  {
    std::vector<std::size_t> const ranks(2, commRank);
    std::vector<std::size_t> const sums(2, commSize * (commSize - 1) / 2);
    // root == 0.
    {
      std::vector<std::size_t> totals;
      mpi::reduce(ranks, &totals, MPI_SUM);
      if (commRank == 0) {
        assert(totals == sums);
      }
    }
    // root == 1.
    {
      int const root = 1 % commSize;
      std::vector<std::size_t> totals;
      mpi::reduce(ranks, &totals, MPI_SUM, MPI_COMM_WORLD, root);
      if (commRank == root) {
        assert(totals == sums);
      }
    }
  }
  
  // All reduce with a built-in type.
  {
    assert(mpi::allReduce(commRank, MPI_SUM) == commSize * (commSize - 1) / 2);
    assert(mpi::allReduceSum(commRank) == commSize * (commSize - 1) / 2);
    assert(mpi::allReduceSum(std::size_t(commRank)) ==
           std::size_t(commSize * (commSize - 1) / 2));
  }
  
  // Broadcast with a built-in type.
  {
    int data;
    if (commRank == 0) {
      data = 7;
    }
    mpi::bcast(&data);
    assert(data == 7);
  }
  
  // Broadcast with a specialized type.
  {
    std::array<int, 3> const value = {{2, 3, 5}};
    std::array<int, 3> data;
    if (commRank == 0) {
      data = value;
    }
    mpi::bcast(&data);
    assert(data == value);
  }
  
  // Broadcast with a POD type.
  {
    typedef Value<int> T;
    int const value = -1;
    T data;
    if (commRank == 0) {
      data.value = value;
    }
    mpi::bcast(&data);
    assert(data.value == value);
  }
  
  // Broadcast a vector with resizing.
  {
    std::vector<int> value;
    value.push_back(2);
    value.push_back(3);
    value.push_back(5);
    std::vector<int> data;
    if (commRank == 0) {
      data = value;
    }
    mpi::bcast(&data);
    assert(data == value);
  }
  
  // Broadcast a vector without resizing.
  {
    std::vector<int> value(3);
    value.push_back(2);
    value.push_back(3);
    value.push_back(5);
    std::vector<int> data(value.size());
    if (commRank == 0) {
      data = value;
    }
    mpi::bcastNoResize(&data);
    assert(data == value);
  }

  // Scatter.
  {
    std::vector<std::size_t> ranks;
    if (commRank == 0) {
      ranks.resize(commSize);
      for (std::size_t i = 0; i != ranks.size(); ++i) {
        ranks[i] = i;
      }
    }
    assert(mpi::scatter(ranks) == std::size_t(commRank));
  }
  
  MPI_Finalize();
  return 0;
}
