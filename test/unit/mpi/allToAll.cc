// -*- C++ -*-

#include "stlib/mpi/allToAll.h"

using namespace stlib;

int
main(int argc, char* argv[])
{
  MPI_Init(&argc, &argv);

  int const commSize = mpi::commSize();
  int const commRank = mpi::commRank();

  //
  // Send a single object.
  //
  {
    std::vector<std::size_t> send(commSize, commRank);
    std::vector<std::size_t> receive(commSize);
    mpi::allToAll(send, &receive);
    assert(receive.size() == send.size());
    for (std::size_t i = 0; i != receive.size(); ++i) {
      assert(receive[i] == i);
    }
  }
  
  //
  // Send and receive PackedArrayOfArrays.
  //
  
  // Send the rank to oneself.
  {
    container::PackedArrayOfArrays<std::size_t> send;
    for (std::size_t i = 0; i != std::size_t(commSize); ++i) {
      send.pushArray();
      if (i == std::size_t(commRank)) {
        send.push_back(i);
      }
    }
    container::PackedArrayOfArrays<std::size_t> receive;
    mpi::allToAll(send, &receive);
    assert(receive.numArrays() == std::size_t(commSize));
    for (std::size_t i = 0; i != receive.numArrays(); ++i) {
      if (i == std::size_t(commRank)) {
        assert(receive.size(i) == 1);
        assert(receive(i, 0) == std::size_t(commRank));
      }
      else {
        assert(receive.size(i) == 0);
      }
    }
  }
  
  // Send each process its own rank.
  {
    container::PackedArrayOfArrays<std::size_t> send;
    for (std::size_t i = 0; i != std::size_t(commSize); ++i) {
      send.pushArray();
      send.push_back(i);
    }
    container::PackedArrayOfArrays<std::size_t> receive;
    mpi::allToAll(send, &receive);
    assert(receive.numArrays() == std::size_t(commSize));
    for (std::size_t i = 0; i != receive.numArrays(); ++i) {
      assert(receive.size(i) == 1);
      assert(receive(i, 0) == std::size_t(commRank));
    }
  }
  
  // Send each process a Cartesian point.
  {
    typedef std::array<float, 3> Point;
    container::PackedArrayOfArrays<Point> send;
    for (std::size_t i = 0; i != std::size_t(commSize); ++i) {
      send.pushArray();
      send.push_back(Point{{float(i), float(i), float(i)}});
    }
    container::PackedArrayOfArrays<Point> receive;
    mpi::allToAll(send, &receive);
    assert(receive.numArrays() == std::size_t(commSize));
    Point const p = {{float(commRank), float(commRank), float(commRank)}};
    for (std::size_t i = 0; i != receive.numArrays(); ++i) {
      assert(receive.size(i) == 1);
      assert(receive(i, 0) == p);
    }
  }
  
  // Send to each process ones' duplicated rank.
  {
    container::PackedArrayOfArrays<std::size_t> send;
    for (std::size_t i = 0; i != std::size_t(commSize); ++i) {
      send.pushArray();
      for (std::size_t j = 0; j != std::size_t(commRank); ++j) {
        send.push_back(commRank);
      }
    }
    container::PackedArrayOfArrays<std::size_t> receive;
    mpi::allToAll(send, &receive);
    assert(receive.numArrays() == std::size_t(commSize));
    for (std::size_t i = 0; i != receive.numArrays(); ++i) {
      assert(receive.size(i) == i);
      for (std::size_t j = 0; j != receive.size(i); ++j) {
        assert(receive(i, j) == i);
      }
    }
  }
  
  //
  // Send PackedArrayOfArrays and receive std::vector.
  //
  
  // Send the rank to oneself.
  {
    container::PackedArrayOfArrays<std::size_t> send;
    for (std::size_t i = 0; i != std::size_t(commSize); ++i) {
      send.pushArray();
      if (i == std::size_t(commRank)) {
        send.push_back(i);
      }
    }
    // Blocking communication.
    {
      std::vector<std::size_t> receive;
      mpi::allToAll(send, &receive);
      assert(receive.size() == 1);
      assert(receive[0] == std::size_t(commRank));
    }
    // Non-blocking communication.
    {
      std::vector<std::size_t> receive;
      std::vector<MPI_Request> sendRequests;
      std::vector<MPI_Request> receiveRequests;
      mpi::allToAll(send, &receive, 0, &sendRequests, &receiveRequests);
      assert(sendRequests.size() == 1);
      assert(receiveRequests.size() == 1);
      mpi::waitAll(&sendRequests);
      mpi::waitAll(&receiveRequests);
      assert(receive.size() == 1);
      assert(receive[0] == std::size_t(commRank));
    }
  }
  
  // Send each process its own rank.
  {
    container::PackedArrayOfArrays<std::size_t> send;
    for (std::size_t i = 0; i != std::size_t(commSize); ++i) {
      send.pushArray();
      send.push_back(i);
    }
    // Blocking communication.
    {
      std::vector<std::size_t> receive;
      mpi::allToAll(send, &receive);
      assert(receive.size() == std::size_t(commSize));
      for (std::size_t i = 0; i != receive.size(); ++i) {
        assert(receive[i] == std::size_t(commRank));
      }
    }
    // Non-blocking communication.
    {
      std::vector<std::size_t> receive;
      std::vector<MPI_Request> sendRequests;
      std::vector<MPI_Request> receiveRequests;
      mpi::allToAll(send, &receive, 0, &sendRequests, &receiveRequests);
      mpi::waitAll(&sendRequests);
      mpi::waitAll(&receiveRequests);
      assert(receive.size() == std::size_t(commSize));
      for (std::size_t i = 0; i != receive.size(); ++i) {
        assert(receive[i] == std::size_t(commRank));
      }
    }
  }
  
  // Send each process a Cartesian point.
  {
    typedef std::array<float, 3> Point;
    container::PackedArrayOfArrays<Point> send;
    for (std::size_t i = 0; i != std::size_t(commSize); ++i) {
      send.pushArray();
      send.push_back(Point{{float(i), float(i), float(i)}});
    }
    // Blocking communication.
    Point const p = {{float(commRank), float(commRank), float(commRank)}};
    {
      std::vector<Point> receive;
      mpi::allToAll(send, &receive);
      assert(receive.size() == std::size_t(commSize));
      for (std::size_t i = 0; i != receive.size(); ++i) {
        assert(receive[i] == p);
      }
    }
    // Non-blocking communication.
    {
      std::vector<Point> receive;
      std::vector<MPI_Request> sendRequests;
      std::vector<MPI_Request> receiveRequests;
      mpi::allToAll(send, &receive, 0, &sendRequests, &receiveRequests);
      mpi::waitAll(&sendRequests);
      mpi::waitAll(&receiveRequests);
      assert(receive.size() == std::size_t(commSize));
      for (std::size_t i = 0; i != receive.size(); ++i) {
        assert(receive[i] == p);
      }
    }
  }
  
  // Send to each process ones' duplicated rank.
  {
    container::PackedArrayOfArrays<std::size_t> send;
    for (std::size_t i = 0; i != std::size_t(commSize); ++i) {
      send.pushArray();
      for (std::size_t j = 0; j != std::size_t(commRank); ++j) {
        send.push_back(commRank);
      }
    }
    // Blocking communication.
    {
      std::vector<std::size_t> receive;
      mpi::allToAll(send, &receive);
      std::size_t n = 0;
      for (std::size_t i = 0; i != std::size_t(commSize); ++i) {
        for (std::size_t j = 0; j != i; ++j) {
          assert(receive[n] == i);
          ++n;
        }
      }
      assert(n == receive.size());
    }
    // Non-blocking communication.
    {
      std::vector<std::size_t> receive;
      std::vector<MPI_Request> sendRequests;
      std::vector<MPI_Request> receiveRequests;
      mpi::allToAll(send, &receive, 0, &sendRequests, &receiveRequests);
      mpi::waitAll(&sendRequests);
      mpi::waitAll(&receiveRequests);
      std::size_t n = 0;
      for (std::size_t i = 0; i != std::size_t(commSize); ++i) {
        for (std::size_t j = 0; j != i; ++j) {
          assert(receive[n] == i);
          ++n;
        }
      }
      assert(n == receive.size());
    }
  }
  
  MPI_Finalize();
  return 0;
}
