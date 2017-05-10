// -*- C++ -*-

#include "geom/orq/KDTree.h"

#include "ads/timer.h"
#include "ads/utility.h"

#include <iostream>
#include <fstream>
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
   typedef std::tr1::array<double, 3> Point;
   typedef std::vector<Point> PointContainer;
   // The record type is a const iterator on points.
   typedef PointContainer::const_iterator Record;
   typedef std::vector<Record> RecordContainer;
   // The ORQ data structure.
   typedef geom::KDTree < 3, Record, Point, double, ads::Dereference<Record>,
           std::back_insert_iterator<RecordContainer> > Orq;
   // Bounding box.
   typedef Orq::BBoxType BBox;
   typedef Orq::SemiOpenIntervalType SemiOpenInterval;
   typedef ads::Timer Timer;

   ads::ParseOptionsArguments parser(argc, argv);

   // Program name.
   programName = parser.getProgramName();

   // If they did not specify the correct number of arguments.
   if (parser.getNumberOfArguments() != 3) {
      std::cerr << "Bad number of required arguments.\n"
                << "You gave the arguments:\n";
      parser.printArguments(std::cerr);
      exitOnError();
   }

   //
   // Parse the arguments.
   //

   //
   // Read the vertices as records from the mesh file.
   //
   PointContainer points;
   {
      // Open the mesh file.
      std::ifstream in(parser.getArgument().c_str());
      int spaceDimension, simplexDimension;
      in >> spaceDimension >> simplexDimension;
      // Check the simplex dimension.
      assert(spaceDimension == 3);
      // We don't care what the simplex dimension is.
      assert(simplexDimension >= 0);
      int numberOfVertices;
      in >> numberOfVertices;
      assert(numberOfVertices > 0);
      points.resize(numberOfVertices);
      for (int i = 0; i != numberOfVertices; ++i) {
         in >> points[i];
      }
      // No need to read the indexed simplices.
   }

   // The search radius for window queries.
   double searchRadius = -1;
   if (! parser.getArgument(&searchRadius)) {
      std::cerr << "Error in reading the search radius.\n";
      exitOnError();
   }
   if (searchRadius < 0) {
      std::cerr << "Bad value for the search radius: " << searchRadius << "\n";
      exitOnError();
   }

   // The leaf size.
   int leafSize = -1;
   if (! parser.getArgument(&leafSize)) {
      std::cerr << "Error in reading the leaf size.\n";
      exitOnError();
   }
   if (leafSize <= 0) {
      std::cerr << "Bad value for the leaf size: " << leafSize << "\n";
      exitOnError();
   }

   std::cout << "SearchRadius = " << searchRadius << "\n"
             << "Leaf size = " << leafSize << "\n";

   //
   // Parse the options.
   //

   const bool areUsingDomain = parser.getOption("domain");

   // There should be no more options.
   assert(parser.areOptionsEmpty());

   // Construct the ORQ data structure.
   Orq orq(points.begin(), points.end(), leafSize);

   std::cout << "Memory usage = " << orq.getMemoryUsage() << "\n";

   //
   // Perform the window queries.
   //

   // Container for records found in a window query.
   std::vector<Record> closePoints;
   // The query window.
   BBox window;
   Point corner;

   Timer timer;
   timer.tic();

   long int count = 0;
   if (areUsingDomain) {
      for (Record i = points.begin(); i != points.end(); ++i) {
         // Set the lower corner of the query window.
         corner = *i;
         corner -= searchRadius;
         window.setLowerCorner(corner);
         // Set the upper corner of the query window.
         corner = *i;
         corner += searchRadius;
         window.setUpperCorner(corner);
         // Perform the window query.
         count += orq.computeWindowQueryUsingDomain
                  (std::back_inserter(closePoints), window);
         // Clear the records in the window query.
         closePoints.clear();
      }
   }
   else {
      for (Record i = points.begin(); i != points.end(); ++i) {
         // Set the lower corner of the query window.
         corner = *i;
         corner -= searchRadius;
         window.setLowerCorner(corner);
         // Set the upper corner of the query window.
         corner = *i;
         corner += searchRadius;
         window.setUpperCorner(corner);
         // Perform the window query.
         count += orq.computeWindowQuery(std::back_inserter(closePoints), window);
         // Clear the records in the window query.
         closePoints.clear();
      }
   }

   const Timer::Number elapsedTime = timer.toc();
   std::cout << "Time = " << elapsedTime << "\n";
   std::cout << "Count = " << count << "\n";

   return 0;
}


void
exitOnError() {
   std::cerr
         << "Bad arguments.  Usage:\n"
         << programName << " [-domain] <mesh> <search radius> <leaf size>\n";
   exit(1);
}
