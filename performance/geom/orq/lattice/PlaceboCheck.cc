// -*- C++ -*-

#include "geom/orq/PlaceboCheck.h"

#include "ads/timer.h"
#include "ads/utility.h"

#include <iostream>
#include <sstream>
#include <vector>

//
// Forward declarations.
//

//! Exit with an error message.
void
exitOnError();

//
// Global variables.
//

static std::string programName;

//! The main loop.
int
main(int argc, char* argv[]) {
   // A Cartesian point.
   typedef ads::FixedArray<3> Point;
   typedef std::vector<Point> PointContainer;
   // The record type is a const iterator on points.
   typedef PointContainer::const_iterator Record;
   // The ORQ data structure.
   typedef geom::PlaceboCheck<3, Record> Orq;
   // Bounding box.
   typedef Orq::BBoxType BBox;
   typedef Orq::SemiOpenIntervalType SemiOpenInterval;
   typedef ads::Timer Timer;

   ads::ParseOptionsArguments parser(argc, argv);

   // Program name.
   programName = parser.getProgramName();

   // If they did not specify the input and output files.
   if (parser.getNumberOfArguments() != 2) {
      std::cerr << "Bad number of required arguments.\n"
                << "You gave the arguments:\n";
      parser.printArguments(std::cerr);
      exitOnError();
   }

   int gridSize;
   {
      std::istringstream gridSizeString(parser.getArgument().c_str());
      gridSizeString >> gridSize;
      assert(gridSize > 1);
   }
   std::cout << "Grid size = " << gridSize << "^3\n";

   double searchRadius;
   {
      std::istringstream searchRadiusString(parser.getArgument().c_str());
      searchRadiusString >> searchRadius;
      assert(searchRadius >= 0);
   }
   std::cout << "Search radius = " << searchRadius << " grid points\n";

   // There should be no more arguments.
   assert(parser.areArgumentsEmpty());

   // The spacing between adjacent grid points.
   const double dx = 1.0 / (gridSize - 1);
   // Radius of the bounding box for a window query.
   const double offset = searchRadius * dx;

   // The points lie on a lattice whose domain spans the unit cube.
   PointContainer file(gridSize * gridSize * gridSize);
   file.clear();
   for (int k = 0; k != gridSize; ++k) {
      for (int j = 0; j != gridSize; ++j) {
         for (int i = 0; i != gridSize; ++i) {
            file.push_back(Point(i * dx, j * dx, k * dx));
         }
      }
   }

   // Construct the ORQ data structure.
   Orq orq(file.begin(), file.end());
   const int diameter = int(std::floor(2 * searchRadius));
   const int querySize = diameter * diameter * diameter;
   orq.setQuerySize(querySize);

   std::cout << "Memory Usage = " << orq.getMemoryUsage() << "\n";

   // Container for records found in a window query.
   std::vector<Record> closePoints;
   // The query window.
   BBox window;
   Timer timer;

   timer.tic();
   long unsigned count = 0;
   // For each point in the lattice.
   for (int k = 0; k < gridSize; ++k) {
      // Set the z-coordinates of the query window.
      window.setLowerCoordinate(2, k * dx - offset);
      window.setUpperCoordinate(2, k * dx + offset);
      for (int j = 0; j < gridSize; ++j) {
         // Set the y-coordinates of the query window.
         window.setLowerCoordinate(1, j * dx - offset);
         window.setUpperCoordinate(1, j * dx + offset);
         for (int i = 0; i < gridSize; ++i) {
            // Set the x-coordinates of the query window.
            window.setLowerCoordinate(0, i * dx - offset);
            window.setUpperCoordinate(0, i * dx + offset);
            // Perform the window query.
            count += orq.computeWindowQuery(std::back_inserter(closePoints),
                                            window);
            // Clear the records in the window query.
            closePoints.clear();
         }
      }
   }

   const Timer::Number elapsedTime = timer.toc();
   std::cout << "time = " << elapsedTime << "\n";
   std::cout << "count = " << count << "\n";

   return 0;
}


void
exitOnError() {
   std::cerr
         << "Bad arguments.  Usage:\n"
         << programName << " <grid size> <search radius>\n";
   exit(1);
}
