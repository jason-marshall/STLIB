// -*- C++ -*-

#include "geom/neighbors/FixedRadiusNeighborSearch.h"

#include "ads/timer/Timer.h"
#include "ads/utility/ParseOptionsArguments.h"
#include "ext/array.h"
#include "numerical/random/uniform/ContinuousUniformGenerator.h"

#include <iostream>

std::string programName;

void
exitOnError() {
   std::cerr << "Usage:\n"
             << programName << " numRecords searchRadius\n";
}

int
main(int argc, char* argv[]) {
   const std::size_t N = 3;
   typedef std::tr1::array<double, N> Point;
   typedef std::vector<Point>::iterator Record;

   ads::ParseOptionsArguments parser(argc, argv);
   programName = parser.getProgramName();

   if (parser.getNumberOfArguments() != 2) {
      std::cerr << "Wrong number of arguments.\n";
      exitOnError();
   }

   std::size_t numRecords = 0;
   if (! parser.getArgument(&numRecords)) {
      std::cerr << "Unable to parse the number of records.\n";
      exitOnError();
   }
   if (numRecords == 0) {
      std::cerr << "The number of records is not allowed to be zero.\n";
      exitOnError();
   }

   double searchRadius = 0;
   if (! parser.getArgument(&searchRadius)) {
      std::cerr << "Unable to parse the search radius.\n";
      exitOnError();
   }
   if (searchRadius <= 0) {
      std::cerr << "The number of search radius must be positive.\n";
      exitOnError();
   }

   // The coordinates.
   std::vector<Point> coordinates(numRecords);
   numerical::ContinuousUniformGeneratorOpen<>::DiscreteUniformGenerator
   generator;
   numerical::ContinuousUniformGeneratorOpen<> random(&generator);
   const double length = std::pow(double(numRecords), 1./3);
   for (std::size_t i = 0; i != coordinates.size(); ++i) {
      for (std::size_t n = 0; n != N; ++n) {
         coordinates[i][n] = length * random();
      }
   }

   std::cout << "Number of records = " << numRecords << ".\n"
             << "Search radius = " << searchRadius << ".\n\n";

   // The search data structure.
   ads::Timer timer;
   timer.tic();
   geom::FixedRadiusNeighborSearch<N, Record>
   neighborSearch(coordinates.begin(), coordinates.end(), searchRadius);
   double elapsedTime = timer.toc();
   std::cout << "Constructed in " << elapsedTime << " seconds.\n"
             << "Construction time per record = "
             << 1e9 * elapsedTime / numRecords << " nanoseconds.\n\n";

   // The queries.
   std::vector<std::size_t> neighbors;
   std::size_t numReported = 0;
   timer.tic();
   for (std::size_t i = 0; i != neighborSearch.getSize(); ++i) {
      neighbors.clear();
      neighborSearch.findNeighbors(std::back_inserter(neighbors), i);
      numReported += neighbors.size();
   }
   elapsedTime = timer.toc();
   std::cout << "Total number of reported records = " << numReported << ".\n"
             << "Total query time = " << elapsedTime << " seconds.\n"
             << "Time per query = " << 1e6 * elapsedTime / numRecords
             << " microseconds.\n"
             << "Time per reported record = "
             << 1e9 * elapsedTime / numReported << " nanoseconds.\n";

   return 0;
}
