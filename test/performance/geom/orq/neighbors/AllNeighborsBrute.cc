// -*- C++ -*-

#include "stlib/ext/array.h"
#include "stlib/ads/utility/ParseOptionsArguments.h"
#include "stlib/ads/timer.h"

#include <vector>

using namespace stlib;

std::string programName;

void
exitOnError()
{
  std::cerr
      << "Bad arguments.  Usage:\n"
      << programName << " numRecords searchRadius\n"
      << "\nExiting...\n";
  exit(1);
}

int
main(int argc, char* argv[])
{
  // Parse the program name and options.
  ads::ParseOptionsArguments parser(argc, argv);
  programName = parser.getProgramName();

  if (parser.getNumberOfArguments() != 2) {
    std::cerr << "Bad number of required arguments.\n"
              << "You gave the arguments:\n";
    parser.printArguments(std::cerr);
    exitOnError();
  }

  std::size_t numRecords;
  parser.getArgument(&numRecords);

  double searchRadius;
  parser.getArgument(&searchRadius);
  const double squaredRadius = searchRadius * searchRadius;

  const std::size_t Dimension = 3;
  const double Length = std::pow(double(numRecords), 1. / 3.);

  typedef std::array<double, Dimension> Value;
  typedef std::vector<Value> ValueContainer;

  // Random points.
  ValueContainer values(numRecords);
  for (std::size_t i = 0; i != values.size(); ++i) {
    for (std::size_t j = 0; j != Dimension; ++j) {
      values[i][j] = rand() * Length / RAND_MAX;
    }
  }

  ads::Timer timer;
  timer.tic();
  std::size_t count = 0;
  for (std::size_t i = 0; i != values.size(); ++i) {
    for (std::size_t j = i + 1; j != values.size(); ++j) {
      if (stlib::ext::squaredDistance(values[i], values[j]) < squaredRadius) {
        count += 2;
      }
    }
  }
  const double elapsedTime = timer.toc();

  std::cout << "Number of neighbors = " << count << '\n'
            << "Average number of neighbors per record = "
            << double(count) / values.size() << '\n'
            << "Elapsed time = " << elapsedTime << " seconds.\n"
            << "Time per neighbor = "
            << elapsedTime / count * 1e9
            << " nanoseconds\n";

  return 0;
}



