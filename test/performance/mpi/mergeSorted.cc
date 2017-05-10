// -*- C++ -*-

#include "stlib/mpi/sort.h"
#include "stlib/mpi/statistics.h"

#include "stlib/performance/SimpleTimer.h"

#if 1
#include "stlib/ads/utility/ParseOptionsArguments.h"
#else
#include <boost/program_options.hpp>
#endif

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
      " [-n=N] [-h]\n"
      "-n: The number of objects per process.\n";
  }
  MPI_Finalize();
  exit(0);
}

int
main(int argc, char* argv[])
{
  // MPI initialization.
  MPI_Init(&argc, &argv);

  try {
    // Parse the options.
#if 1
    stlib::ads::ParseOptionsArguments parser(argc, argv);
    programName = parser.getProgramName();
    if (parser.getOption('h')) {
      helpMessage();
    }
    std::size_t size = 1024;
    parser.getOption('n', &size);
    if (size == 0) {
        std::cout << "Error: You must specify a postive number of objects.\n";
        helpMessage();
    }
#else
    std::size_t size = 0;
    {
      namespace po = boost::program_options;
      po::options_description desc("Allowed options");
      desc.add_options()
        ("help,h", "produce help message")
        ("number,n", po::value<std::size_t>(), "number of objects per process")
        ;

      po::variables_map vm;
      po::store(po::parse_command_line(argc, argv, desc), vm);
      po::notify(vm);

      if (vm.count("help")) {
        std::cout << desc << "\n";
        return 0;
      }

      if (! vm.count("number")) {
        std::cout << "Error: You must specify the number of objects.\n";
        return 1;
      }
      size = vm["number"].as<std::size_t>();
    }
#endif
    
    int const commSize = stlib::mpi::commSize();
    int const commRank = stlib::mpi::commRank();

    // The values in each process don't overlap.
    std::default_random_engine generator;
    std::size_t const NumBins = 1 << 20;
    std::uniform_int_distribution<std::size_t> distribution(0, NumBins);
    std::vector<std::size_t> objects(size);
    for (std::size_t i = 0; i != objects.size(); ++i) {
      objects[i] = commRank * NumBins + distribution(generator);
    }
    std::vector<std::size_t> sorted;

    stlib::performance::SimpleTimer timer;
    MPI_Barrier(MPI_COMM_WORLD);
    timer.start();
    std::sort(objects.begin(), objects.end());
#if 0
    stlib::mpi::mergeSortedBinarySearch(objects, &sorted);
#else
    stlib::mpi::mergeSortedSequentialScan(objects, &sorted);
#endif
    timer.stop();

    if (commRank == 0) {
      if (! sorted.empty()) {
        std::cout << "Meaningless result = " << sorted.back() << '\n';
      }
    }
    stlib::mpi::printStatistics(std::cout, "Total time (s)", timer.elapsed());
    if (! objects.empty()) {
      stlib::mpi::printStatistics(std::cout, "Time per element (ns)", 
                                  timer.nanoseconds() /
                                  double(commSize * objects.size()));
    }
  }
  catch(std::exception const& e) {
    std::cerr << "error: " << e.what() << "\n";
    return 1;
  }
  catch(...) {
    std::cerr << "Exception of unknown type!\n";
    return 1;
  }

  MPI_Finalize();
  return 0;
}
