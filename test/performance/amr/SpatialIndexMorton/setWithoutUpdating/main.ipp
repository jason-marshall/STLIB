// -*- C++ -*-

#ifndef __main_ipp__
#error This is an implementation detail.
#endif

#include "stlib/amr/SpatialIndexMorton.h"

#include "stlib/ads/timer/Timer.h"
#include "stlib/ads/utility/ParseOptionsArguments.h"

#include <iostream>

#include <cassert>

using namespace stlib;

namespace {

//! The program name.
static std::string programName;

//! Exit with an error message.
void
exitOnError() {
   std::cerr
         << "Bad arguments.  Usage:\n"
         << programName << "\n";
   exit(1);
}
}

//! The main loop.
int
main(int argc, char* argv[]) {
   const int Dimension = 3;

   typedef amr::SpatialIndexMorton<Dimension, MaximumLevel> SpatialIndexMorton;
   typedef SpatialIndexMorton::Level Level;
   typedef SpatialIndexMorton::Coordinate Coordinate;
   typedef SpatialIndexMorton::CoordinateList CoordinateList;

   ads::ParseOptionsArguments parser(argc, argv);

   // Program name.
   programName = parser.getProgramName();

   if (parser.getNumberOfArguments() != 0) {
      std::cerr << "Bad number of required arguments.\n"
                << "You gave the arguments:\n";
      parser.printArguments(std::cerr);
      exitOnError();
   }

   // There should be no arguments.
   assert(parser.areArgumentsEmpty());

   // There should be no options.
   if (! parser.areOptionsEmpty()) {
      std::cerr << "Error.  Unmatched options:\n";
      parser.printOptions(std::cerr);
      exitOnError();
   }

   ads::Timer timer;
   double elapsedTime;
   unsigned long count = 1000000;
   Coordinate result = 0;
   do {
      SpatialIndexMorton spatialIndex;
      CoordinateList coordinates;
      count *= 2;
      timer.tic();
      unsigned long c = 0;
      while (c != count) {
         for (Level level = 0; level <= MaximumLevel && c != count; ++level) {
            //coordinates.fill(0);
            std::fill(coordinates.begin(), coordinates.end(), 0);
            for (unsigned c0 = 0; c0 != (1U << level) && c != count; ++c0) {
               coordinates[0] = c0;
               for (unsigned c1 = 0; c1 != (1U << level) && c != count; ++c1) {
                  coordinates[1] = c1;
                  for (unsigned c2 = 0; c2 != (1U << level) && c != count; ++c2) {
                     coordinates[2] = c2;
                     spatialIndex.setWithoutUpdating(level, coordinates);
                     result += spatialIndex.getLevel();
                     result += spatialIndex.getCoordinates()[0];
                     result += spatialIndex.getCoordinates()[1];
                     result += spatialIndex.getCoordinates()[2];
                     ++c;
                  }
               }
            }
         }
      }
      elapsedTime = timer.toc();
   }
   while (elapsedTime < 1);

   std::cout << "Meaningless result = " << result << "\n"
             << "Dimension = " << Dimension << "\n"
             << "MaximumLevel = " << MaximumLevel << "\n"
             << count << " operations in " << elapsedTime << " seconds.\n"
             << "Time per operation = "
             << elapsedTime / count * 1e9
             << " nanoseconds.\n";
   std::cout.setf(std::ios_base::fixed, std::ios_base::floatfield);
   std::cout.precision(0);
   std::cout << elapsedTime / count * 1e9 << "\n";

   return 0;
}

