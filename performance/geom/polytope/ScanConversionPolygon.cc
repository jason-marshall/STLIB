// -*- C++ -*-

#include "geom/polytope/ScanConversionPolygon.h"

#include "ads/timer/Timer.h"
#include "ads/utility/ParseOptionsArguments.h"

#include <iostream>
#include <fstream>

int
main(int argc, char* argv[]) {
   typedef geom::ScanConversionPolygon<> Polygon;
   typedef ads::FixedArray<2, int> MultiIndex;

   ads::ParseOptionsArguments parser(argc, argv);

   // Check the argument count.
   if (parser.getNumberOfArguments() != 1) {
      std::cerr << "Error: You must specify the polygon.\n";
      exit(1);
   }

   // Get the polygon.
   Polygon polygon;
   {
      std::ifstream in(parser.getArgument().c_str());
      in >> polygon;
   }
   if (polygon.getVerticesSize() < 1) {
      std::cerr << "Error: Bad polygon.\n";
      exit(1);
   }

   // Compute suitable grid extents.
   MultiIndex extents;
   {
      geom::BBox<2> domain;
      domain.bound(polygon.getVerticesBeginning(), polygon.getVerticesEnd());
      extents[0] = std::max(1, int(domain.getUpperCorner()[0]) + 1);
      extents[1] = std::max(1, int(domain.getUpperCorner()[1]) + 1);
      assert(extents[0] > 0 && extents[1] > 0);
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
         polygon.scanConvert(std::back_inserter(indices), extents);
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
