// -*- C++ -*-

#ifndef __OrthtreeMap_ipp__
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
   // Make the refinement and coarsening criteria return true.
   orthtree.getRefine().set(true);
   orthtree.getCoarsen().set(true);
   // Insert a single leaf.
   orthtree.insert(Key());
   // Refine to get a full tree.
   orthtree.refine();
   Orthtree copy(orthtree);

   int coarsenCount = 0;
   ads::Timer timer;
   double elapsedTime;
   int count = 1;
   do {
      count *= 2;
      timer.tic();
      for (int n = 0; n != count; ++n) {
         orthtree = copy;
         coarsenCount += orthtree.coarsen();
      }
      elapsedTime = timer.toc();
   }
   while (elapsedTime < 1);

   std::cout << "Number of nodes = " << copy.size() << "\n"
             << "Time = " << elapsedTime << " seconds.\n"
             << "Time per split = "
             << elapsedTime / coarsenCount * 1e9
             << " nanoseconds.\n";

   return 0;
}

// End of file.

