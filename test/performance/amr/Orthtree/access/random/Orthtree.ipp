// -*- C++ -*-

#ifndef __performance_Orthtree_ipp__
#error This is an implementation detail.
#endif

#include "stlib/amr/Orthtree.h"
#include "stlib/amr/Traits.h"
#include "stlib/amr/CellData.h"
#include "stlib/amr/Patch.h"

#include "stlib/ads/timer/Timer.h"
#include "stlib/ads/utility/ParseOptionsArguments.h"
#include "stlib/ads/functor/constant.h"

#include <algorithm>
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
   typedef amr::Traits<Dimension, MaximumLevel> Traits;
   typedef Traits::SizeList SizeList;
   typedef Traits::SpatialIndex SpatialIndex;
   typedef Traits::Point Point;
   typedef Traits::Number Number;

   typedef amr::CellData<Traits, 1U, 0U> CellData;
   typedef amr::Patch<CellData, Traits> Patch;
   typedef amr::Orthtree<Patch, Traits> Orthtree;
   typedef Orthtree::iterator iterator;
   typedef Orthtree::const_iterator const_iterator;

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
   Orthtree orthtree(ext::filled_array<Point>(0.),
                     ext::filled_array<Point>(1.));
   // Insert a single leaf.
   {
      SpatialIndex key;
      orthtree.insert(key, Patch(key, ext::filled_array<SizeList>(1U)));
   }
   // Refine to get a full tree.
   refine(&orthtree, ads::constructUnaryConstant<iterator>(true));
   // Set the elements.
   {
      int rank = 0;
      for (iterator i = orthtree.begin(); i != orthtree.end(); ++i) {
         // Use container indexing in the multi-array.
         i->second.getPatchData().getArray()[0][0] = rank++;
      }
   }
   // Get the keys.
   std::vector<SpatialIndex> keys;
   for (const_iterator i = orthtree.begin(); i != orthtree.end(); ++i) {
      keys.push_back(i->first);
   }
   // Shuffle the keys.
   std::random_shuffle(keys.begin(), keys.end());

   Number result = 0;
   ads::Timer timer;
   double elapsedTime;
   int count = 1;
   const int size = keys.size();
   do {
      count *= 2;
      timer.tic();
      for (int n = 0; n != count; ++n) {
         for (int i = 0; i != size; ++i) {
            result +=
               orthtree.find(keys[i])->second.getPatchData().getArray()[0][0];
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
   std::cout.setf(std::ios_base::fixed, std::ios_base::floatfield);
   std::cout.precision(0);
   std::cout << elapsedTime / (orthtree.size() * count) * 1e9 << "\n";

   return 0;
}
