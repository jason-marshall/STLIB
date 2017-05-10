// -*- C++ -*-

#include "stlib/geom/orq/CellArrayAllNeighbors.h"
#include "stlib/ads/utility/ParseOptionsArguments.h"
#include "stlib/ads/timer.h"

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

  const std::size_t Dimension = 3;
  const double Length = std::pow(double(numRecords), 1. / 3.);

  typedef std::array<double, Dimension> Value;
  typedef std::vector<Value> ValueContainer;
  typedef ValueContainer::const_iterator Record;
  typedef geom::CellArrayAllNeighbors<Dimension, Record> NS;

  // Random points.
  ValueContainer values(numRecords);
  for (std::size_t i = 0; i != values.size(); ++i) {
    for (std::size_t j = 0; j != Dimension; ++j) {
      values[i][j] = rand() * Length / RAND_MAX;
    }
  }

  NS ns(searchRadius);
  // Warm up.
  ns.allNeighbors(values.begin(), values.end());

  ads::Timer timer;
  timer.tic();
  ns.allNeighbors(values.begin(), values.end());
  const double elapsedTime = timer.toc();

  std::cout << "Number of neighbors = " << ns.packedNeighbors.size() << '\n'
            << "Average number of neighbors per record = "
            << double(ns.packedNeighbors.size()) / values.size() << '\n'
            << "Elapsed time = " << elapsedTime << " seconds.\n"
            << "Time per neighbor = "
            << elapsedTime / ns.packedNeighbors.size() * 1e9
            << " nanoseconds\n";

  return 0;
}



