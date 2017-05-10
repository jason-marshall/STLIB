// -*- C++ -*-

#include "geom/polytope/ScanConversionPolyhedron.h"

#include "ads/timer/Timer.h"
#include "ads/utility/ParseOptionsArguments.h"

#include <iostream>
#include <fstream>

int
main(int argc, char* argv[]) {
   typedef geom::ScanConversionPolyhedron<> Polyhedron;
   typedef ads::FixedArray<3, int> MultiIndex;

   ads::ParseOptionsArguments parser(argc, argv);

   // Check the argument count.
   if (parser.getNumberOfArguments() != 1) {
      std::cerr << "Error: You must specify the polyhedron.\n";
      exit(1);
   }

   // Get the polyhedron.
   Polyhedron polyhedron;
   {
      std::ifstream in(parser.getArgument().c_str());
      in >> polyhedron;
   }
   if (polyhedron.getEdges().size() < 1) {
      std::cerr << "Error: Bad polyhedron.\n";
      exit(1);
   }

   // Determine a suitable grid.
   geom::RegularGrid<3> grid;
   {
      geom::BBox<3> polyhedronDomain;
      polyhedron.computeBBox(&polyhedronDomain);
      MultiIndex extents;
      extents[0] = std::max(2, int(polyhedronDomain.getUpperCorner()[0]) + 1);
      extents[1] = std::max(2, int(polyhedronDomain.getUpperCorner()[1]) + 1);
      extents[2] = std::max(2, int(polyhedronDomain.getUpperCorner()[2]) + 1);
      geom::BBox<3> gridDomain(0, 0, 0,
                               extents[0] - 1, extents[1] - 1, extents[2] - 1);
      grid = geom::RegularGrid<3>(extents, gridDomain);
   }

   // There should be no more arguments.
   if (! parser.areArgumentsEmpty()) {
      std::cerr << "Error: Un-parsed arguments:\n";
      parser.printArguments(std::cerr);
      exit(1);
   }

   // Get the target time per test.  The default is 1 second.
   double timePerTest = 1;
   parser.getOption("time", &timePerTest) ||
   parser.getOption("t", &timePerTest);
   if (timePerTest <= 0) {
      std::cerr << "Error: Bad target time per test.\n";
      exit(1);
   }

   // There should be no more options.
   if (! parser.areOptionsEmpty()) {
      std::cerr << "Error: Unmatched options.\n";
      parser.printOptions(std::cerr);
      exit(1);
   }

   //
   // Determine an appropriate number of times to evaluate the functor.
   // Test the functor.
   //
   ads::Timer timer;
   int count = 1;
   double time;
   std::vector<MultiIndex> indices;
   do {
      count *= 2;
      timer.tic();
      for (int n = 0; n != count; ++n) {
         indices.clear();
         polyhedron.scanConvert(std::back_inserter(indices), grid);
      }
      time = timer.toc();
   }
   while (time < timePerTest);

   // Print the time per call in nanoseconds.
   std::cout
         << "Time per grid point = " << timer.toc() / count * 1e9 / indices.size()
         << " nanoseconds.\n"
         << "Time per scan conversion = " << timer.toc() / count
         << " seconds.\n"
         << "Number of tests = " << count << "\n"
         << "Number of grid points per scan conversion = "
         << indices.size() << "\n";

   return 0;
}
