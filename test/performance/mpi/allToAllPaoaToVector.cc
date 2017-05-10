// -*- C++ -*-

#include "stlib/mpi/allToAll.h"
#include "stlib/mpi/statistics.h"

#include "stlib/ads/utility/ParseOptionsArguments.h"
#include "stlib/numerical/partition.h"
#include "stlib/performance/SimpleTimer.h"

#include <random>

// The program name.
std::string programName;

// Exit with a usage message.
void
helpMessage()
{
  if (stlib::mpi::commRank() == 0) {
    std::cout
      << "Usage:\n"
      << programName <<
      " [-o=O] [-s=S] [-n] [-h]\n"
      "-o: The number of objects per process.\n"
      "-s: The number of processes to send to.\n"
      "-n: Use non-blocking communication.\n";
  }
  MPI_Finalize();
  exit(0);
}

int
main(int argc, char* argv[])
{
  // The object type is a triangle.
  typedef double Float;
  std::size_t const Dimension = 3;
  typedef std::array<Float, Dimension> Point;
  typedef std::array<Point, 3> Object;

  // MPI initialization.
  MPI_Init(&argc, &argv);

  int const commSize = stlib::mpi::commSize();
  int const commRank = stlib::mpi::commRank();

  try {
    // Parse the options.
    std::size_t numObjects = 0;
    std::size_t numSendProcesses = 0;
    bool useNonBlocking = false;
    {
      stlib::ads::ParseOptionsArguments parser(argc, argv);
      programName = parser.getProgramName();
      if (parser.getOption('h')) {
        helpMessage();
      }

      parser.getOption('o', &numObjects);
      if (numObjects == 0) {
        std::cout << "Error: The number of objects must be positive.\n";
        helpMessage();
      }

      parser.getOption('s', &numSendProcesses);
      if (numSendProcesses == 0) {
        std::cout << "Error: The number of send processes must be positive.\n";
        helpMessage();
      }

      useNonBlocking = parser.getOption('n');
    }
    if (commRank == 0) {
      std::cout << "Send " << numObjects << " objects to " << numSendProcesses
                << " processes.\n"
                << "Size of object is " << sizeof(Object) << " bytes.\n";
      if (useNonBlocking) {
        std::cout << "Use non-blocking communication.\n";
      }
      else {
        std::cout << "Use blocking communication.\n";
      }
    }

    // Calculate a partitioning of the objects.
    // Start with delimiters.
    std::vector<std::size_t> delimiters;
    stlib::numerical::computePartitions(numObjects, numSendProcesses,
                                        std::back_inserter(delimiters));
    // Convert to sizes.
    std::vector<std::size_t> sendSizes(delimiters.size() - 1);
    for (std::size_t i = 0; i != sendSizes.size(); ++i) {
      sendSizes[i] = delimiters[i + 1] - delimiters[i];
    }
    // Convert to the sizes that we will send.
    std::vector<std::size_t> sizes(commSize, 0);
    for (std::size_t i = 0; i != sendSizes.size(); ++i) {
      sizes[(commRank + i) % commSize] = sendSizes[i];
    }

    // Make the container of objects to send. (The actual values don't matter.
    // so we leave them uninitialized).
    stlib::container::PackedArrayOfArrays<Object> send;
    send.rebuild(sizes.begin(), sizes.end());

    std::vector<Object> receive;
    stlib::performance::SimpleTimer timer;
    MPI_Barrier(MPI_COMM_WORLD);
    timer.start();
    if (useNonBlocking) {
      std::vector<MPI_Request> sendRequests;
      std::vector<MPI_Request> receiveRequests;
      stlib::mpi::allToAll(send, &receive, 0, &sendRequests, &receiveRequests);
      stlib::mpi::waitAll(&sendRequests);
      stlib::mpi::waitAll(&receiveRequests);
    }
    else {
      stlib::mpi::allToAll(send, &receive);
    }
    timer.stop();

    if (commRank == 0) {
      std::cout << "Meaningless result = " << receive.back()[0][0] << '\n';
    }
    stlib::mpi::printStatistics(std::cout, "Total time (s)", timer.elapsed());
    stlib::mpi::printStatistics(std::cout, "Time per byte per process (ns)", 
                                timer.nanoseconds() /
                                double(send.size() * sizeof(Object)));
  }
  catch(std::exception const& e) {
    std::cerr << "error: " << e.what() << "\n";
    MPI_Finalize();
    return 1;
  }
  catch(...) {
    std::cerr << "Exception of unknown type!\n";
    MPI_Finalize();
    return 1;
  }

  MPI_Finalize();
  return 0;
}
