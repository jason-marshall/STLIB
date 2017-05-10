// -*- C++ -*-

/*
  rcb3D.cc

  Recursive coordinate bisection example in 3-D.
*/

#include "stlib/concurrent/partition/rcb.h"
#include "stlib/ads/utility/ParseOptionsArguments.h"

#include <iostream>

#include <cassert>
#include <cstdlib>

int
main(int argc, char* argv[]) {
  // The dimension.
  const std::size_t N = 3;
  typedef std::array<double, N> Point;

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
  for (std::size_t i = 0; i != positions.size(); ++i) {
    for (std::size_t n = 0; n != positions[i].size(); ++n) {
      positions[i][n] = double(rand()) / RAND_MAX;
    }
  }

  //
  // Partition the data.
  //

  stlib::concurrent::rcb<N>(numProcessors, &identifiers, &idPartition,
                            positions);

  //
  // Check the results.
  //

  // A bounding box for each processor.
  typedef stlib::geom::BBox<double, N> BBox;
  std::vector<BBox> intervals(numProcessors);

  std::cout << '\n';
  for (std::size_t p = 0; p != numProcessors; ++p) {
    std::cout << "Processor " << p << " has "
              << idPartition[p+1] - idPartition[p] << " records.\n";
    intervals[p] = stlib::geom::specificBBox<BBox>
      (&positions[*idPartition[p]], &positions[*idPartition[p + 1]]);
    std::cout << "Domain = " << intervals[p] << '\n';
  }

#if 0
  // Check that the open intervals do not overlap.
  for (std::size_t m = 0; m < numProcessors - 1; ++m) {
    for (std::size_t n = m + 1; n < numProcessors; ++n) {
      assert(! doOverlap(intervals[m], intervals[n]));
    }
  }
#endif

  return 0;
}
