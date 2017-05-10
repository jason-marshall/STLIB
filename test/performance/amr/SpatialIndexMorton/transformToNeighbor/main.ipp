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
   typedef SpatialIndexMorton::Code Code;

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
   Code code = 0;
   do {
      SpatialIndexMorton spatialIndex;
      spatialIndex.setLevelWithoutUpdating(MaximumLevel);
      spatialIndex.transformToNeighbor(1);
      spatialIndex.transformToNeighbor(3);
      spatialIndex.transformToNeighbor(5);
      count *= 2;
      timer.tic();
      unsigned long c = 0;
      while (c != count) {
         for (int d = 0; d != 2 * Dimension && c != count; ++d) {
            spatialIndex.transformToNeighbor(d);
            code += spatialIndex.getCode();
            ++c;
         }
      }
      elapsedTime = timer.toc();
   }
   while (elapsedTime < 1);

   std::cout << "Meaningless result = " << code << "\n"
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

