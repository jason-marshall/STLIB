// -*- C++ -*-

#include "geom/orq/Octree.h"

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
   typedef ads::FixedArray<3> Point;
   typedef std::vector<Point> PointContainer;
   // The record type is a const iterator on points.
   typedef PointContainer::const_iterator Record;
   typedef std::vector<Record> RecordContainer;
   // The ORQ data structure.
   typedef geom::Octree < Record, Point, double, ads::Dereference<Record>,
           std::back_insert_iterator<RecordContainer> > Orq;
   // Bounding box.
   typedef Orq::BBox BBox;
   typedef Orq::SemiOpenInterval SemiOpenInterval;
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

   //
   // Read the window queries file.
   //
   std::vector<BBox> queries;
   int numberOfTests = -1, numberOfQueriesPerTest = -1;
   {
      // Open the queries file.
      std::ifstream in(parser.getArgument().c_str());

      in >> numberOfTests >> numberOfQueriesPerTest;
      assert(numberOfTests >= 0);
      assert(numberOfQueriesPerTest >= 0);
      const int Size = numberOfTests * numberOfQueriesPerTest;
      queries.resize(Size);

      BBox window;
      for (int i = 0; i != Size; ++i) {
         in >> queries[i];
      }
   }
   std::cout << "Number of tests = " << numberOfTests
             << "\nNumbof of queries per test = " << numberOfQueriesPerTest
             << "\n";

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
   std::cout << "Leaf size = " << leafSize << "\n";

   //
   // Find the domain.
   //
   SemiOpenInterval domain;
   // Bound the vertices.
   domain.bound(points.begin(), points.end());
   {
      // Slightly enlarge the domain.
      Point diagonal = domain.getUpperCorner() - domain.getLowerCorner();
      const double Epsilon = std::sqrt(std::numeric_limits<double>::epsilon());
      domain.setLowerCorner(domain.getLowerCorner() - Epsilon * diagonal);
      domain.setUpperCorner(domain.getUpperCorner() + Epsilon * diagonal);
   }

   // Construct the ORQ data structure.
   Orq orq(domain, points.begin(), points.end(), leafSize);

   std::cout << "Memory usage = " << orq.getMemoryUsage() << "\n";

   //
   // Perform the window queries.
   //

   // Container for records found in a window query.
   std::vector<Record> closePoints;
   int count;
   Timer timer;
   Timer::Number elapsedTime;

   // Warm up.
   for (int j = 0; j != numberOfQueriesPerTest; ++j) {
      count = orq.computeWindowQuery(std::back_inserter(closePoints), queries[j]);
      // Clear the records in the window query.
      closePoints.clear();
   }

   int n = 0;
   for (int i = 0; i != numberOfTests; ++i) {
      timer.tic();
      for (int j = 0; j != numberOfQueriesPerTest; ++j) {
         count = orq.computeWindowQuery(std::back_inserter(closePoints),
                                        queries[j]);
         // Clear the records in the window query.
         closePoints.clear();
         ++n;
      }
      elapsedTime = timer.toc();
      std::cout << (elapsedTime / numberOfQueriesPerTest) << ",\n";
   }
   std::cout << "\n";

   return 0;
}


void
exitOnError() {
   std::cerr
         << "Bad arguments.  Usage:\n"
         << programName << " <mesh> <queries> <leafSize>\n";
   exit(1);
}
