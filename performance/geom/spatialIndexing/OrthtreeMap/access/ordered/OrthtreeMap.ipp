// -*- C++ -*-

#ifndef __performance_geom_spatialIndexing_OrthtreeMap_access_ordered_ipp__
#error This is an implementation detail.
#endif

#include "geom/spatialIndexing/OrthtreeMap.h"

#include "ads/timer/Timer.h"
#include "ads/utility/ParseOptionsArguments.h"

#include <iostream>

#include <cassert>

namespace {

//
// Types.
//

//! The number type.
typedef double Number;

//
// Orthtree types.
//

typedef geom::OrthtreeMap<Dimension, MaximumLevel, Number> Orthtree;
typedef Orthtree::Point Point;
typedef Orthtree::Key Key;
typedef Orthtree::Element Element;
typedef Orthtree::iterator iterator;
typedef Orthtree::const_iterator const_iterator;

//
// Global variables.
//

//! The program name.
static std::string programName;

//
// Error message.
//

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

   // Construct the orthtree.
   Orthtree orthtree(Point(0.), Point(1.));
   // Make the refinement criterion return true.
   orthtree.getRefine().set(true);
   // Insert a single leaf.
   orthtree.insert(Key());
   // Refine to get a full tree.
   orthtree.refine();
   // Set the elements.
   {
      int rank = 0;
      for (iterator i = orthtree.begin(); i != orthtree.end(); ++i) {
         i->second = rank++;
      }
   }

   Element result = 0;
   ads::Timer timer;
   double elapsedTime;
   int count = 1;
   do {
      count *= 2;
      timer.tic();
      for (int n = 0; n != count; ++n) {
         for (const_iterator i = orthtree.begin(); i != orthtree.end(); ++i) {
            result += i->second;
         }
      }
      elapsedTime = timer.toc();
   }
   while (elapsedTime < 1);

   std::cout << "Meaningless result = " << result << "\n"
             << "Number of nodes = " << orthtree.size() << "\n"
             << "Time = " << elapsedTime << " seconds.\n"
             << "Time per access = "
             << elapsedTime / (orthtree.size() * count) * 1e9
             << " nanoseconds.\n";

   return 0;
}

// End of file.

