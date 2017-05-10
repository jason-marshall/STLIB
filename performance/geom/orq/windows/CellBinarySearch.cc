// -*- C++ -*-

#include "geom/orq/CellBinarySearch.h"

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
   // The ORQ data structure.
   typedef geom::CellBinarySearch<3, Record> Orq;
   // Bounding box.
   typedef Orq::BBox BBox;
   typedef Orq::SemiOpenInterval SemiOpenInterval;
   typedef ads::Timer Timer;

   ads::ParseOptionsArguments parser(argc, argv);

   // Program name.
   programName = parser.getProgramName();

   // If they did not specify the correct number of arguments.
   if (parser.getNumberOfArguments() != 5) {
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

   // The size of a cell.
   Point cellSize(-1, -1, -1);
   if (!(parser.getArgument(&cellSize[0]) &&
         parser.getArgument(&cellSize[1]))) {
      std::cerr << "Error in reading the cell size.\n";
      exitOnError();
   }
   if (cellSize[0] <= 0 || cellSize[1] <= 0) {
      std::cerr << "Bad value for the cell size: " << cellSize[0] << " "
                << cellSize[1] << "\n";
      exitOnError();
   }
   std::cout << "Cell size = " << cellSize[0] << " " << cellSize[1] << "\n";


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

   // Construct the cell array.
   Orq orq(cellSize, domain, points.begin(), points.end());

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
         << programName << " <mesh> <queries> <dx> <dy>\n";
   exit(1);
}
