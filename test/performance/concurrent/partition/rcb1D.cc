// -*- C++ -*-

/*
  rcb1D.cc

  Recursive coordinate bisection example in 1-D.
*/

#include "stlib/concurrent/partition/rcb.h"
#include "stlib/ads/utility/ParseOptionsArguments.h"

#include <iostream>

#include <cassert>
#include <cstdlib>

USING_STLIB_EXT_ARRAY_IO_OPERATORS;

int
main(int argc, char* argv[]) {
  typedef std::array<double, 1> Point;

  //
  // Process the command line arguments.
  //
  stlib::ads::ParseOptionsArguments parser(argc, argv);

  // Check the number of arguments.
  if (parser.getNumberOfArguments() != 2) {
    std::cerr << "\nError: Bad arguments.\n"
              << "Usage:\n"
              << parser.getProgramName() << " numProcessors numRecords\n";
    exit(1);
  }

  // Get the number of processors.
  std::size_t numProcessors = 0;
  parser.getArgument(&numProcessors);
  assert(numProcessors > 0);

  // Get the number of records.
  std::size_t numRecords = 0;
  parser.getArgument(&numRecords);

  //
  // Make the data.
  //

  std::vector<std::size_t> identifiers(numRecords);
  // Initialize the identifiers.
  for (std::size_t n = 0; n != identifiers.size(); ++n) {
    identifiers[n] = n;
  }
  std::vector<std::size_t*> idPartition(numProcessors + 1);
  std::vector<Point> positions(numRecords);
  // Set the positions.
  for (std::size_t n = 0; n != positions.size(); ++n) {
    positions[n][0] = double(rand()) / RAND_MAX;
  }

  //
  // Partition the data.
  //

  stlib::concurrent::rcb<1>(numProcessors, &identifiers, &idPartition,
                            positions);

  //
  // Print the results.
  //

  std::cout << '\n';
  for (std::size_t p = 0; p != numProcessors; ++p) {
    std::cout << "Processor " << p << " has "
              << idPartition[p+1] - idPartition[p] << " records.\n";
    std::copy(idPartition[p], idPartition[p+1],
              std::ostream_iterator<std::size_t>(std::cout, " "));
    std::cout << '\n';
    for (std::size_t* i = idPartition[p]; i != idPartition[p+1]; ++i) {
      std::cout << positions[*i] << " ";
    }
    std::cout << '\n';
  }

  return 0;
}
