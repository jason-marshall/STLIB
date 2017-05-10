// -*- C++ -*-

#ifndef __ads_IndexedPriorityQueue_main_ipp__
#error This file is an implementation detail.
#endif

#include "stlib/ads/timer/Timer.h"
#include "stlib/ads/utility/ParseOptionsArguments.h"
#include "stlib/ext/vector.h"

#include <algorithm>
#include <iostream>
#include <fstream>

namespace {

//
// Types.
//

typedef double Number;

//
// Global variables.
//

//! The program name.
static std::string programName;

//
// Local functions.
//

//! Exit with an error message.
void
exitOnError() {
   std::cerr
         << "Bad arguments.  Usage:\n"
         << programName
#ifdef HASHING
         << " [-count=integer] [-table=integer] [-load=real]"
#endif
         << " propensities\n";
   exit(1);
}

template<typename _IndexedPriorityQueue>
inline
void
setPropensities(std::false_type /*UsesPropensities*/,
                _IndexedPriorityQueue* /*indexedPriorityQueue*/,
                std::vector<Number>* /*propensities*/) {
}

template<typename _IndexedPriorityQueue>
inline
void
setPropensities(std::true_type /*UsesPropensities*/,
                _IndexedPriorityQueue* indexedPriorityQueue,
                std::vector<Number>* propensities) {
   indexedPriorityQueue->setPropensities(propensities);
}

}


//! The main loop.
int
main(int argc, char* argv[]) {
  USING_STLIB_EXT_VECTOR_IO_OPERATORS;

   ads::ParseOptionsArguments parser(argc, argv);

   // Program name.
   programName = parser.getProgramName();

   if (parser.getNumberOfArguments() != 1) {
      std::cerr << "Bad number of required arguments.\n"
                << "You gave the arguments:\n";
      parser.printArguments(std::cerr);
      exitOnError();
   }

   // Read the propensities.
   std::vector<Number> propensities;
   {
      std::ifstream in(parser.getArgument().c_str());
      in >> propensities;
      if (propensities.size() < 2) {
         std::cerr << "Bad size for the propensities array.";
         exitOnError();
      }
   }
   std::random_shuffle(propensities.begin(), propensities.end());

   // There should be no more arguments.
   assert(parser.areArgumentsEmpty());

   long count = 0;
   if (parser.getOption("count", &count)) {
      if (count < 1) {
         std::cerr << "Bad count = " << count << "\n";
         exitOnError();
      }
   }

   std::vector<Number> offsets(propensities.size());
   for (std::size_t i = 0; i != offsets.size(); ++i) {
      assert(propensities[i] > 0);
      offsets[i] = 1 / propensities[i];
   }

   const Number sumPropensities = ext::sum(propensities);
   assert(sumPropensities > 0);
   const Number factor = 1 / sumPropensities;
   std::vector<Number> initialTimes(propensities.size());
   for (std::size_t i = 0; i != offsets.size(); ++i) {
      initialTimes[i] = i * factor;
   }
   std::random_shuffle(initialTimes.begin(), initialTimes.end());
   std::vector<Number> times(propensities.size());

   // Construct the indexed priority queue.
#ifdef HASHING
   int tableSize = std::max(std::size_t(256), 2 * propensities.size());
   parser.getOption("table", &tableSize);
   if (tableSize < 1) {
      std::cerr << "Bad size for the hash table = " << tableSize << "\n";
      exitOnError();
   }

   Number targetLoad = 2;
   parser.getOption("load", &targetLoad);
   if (targetLoad <= 0) {
      std::cerr << "Bad target load for the hash table = " << targetLoad << "\n";
      exitOnError();
   }

   IndexedPriorityQueue indexedPriorityQueue(propensities.size(), tableSize,
         targetLoad);
#else
   IndexedPriorityQueue indexedPriorityQueue(propensities.size());
#endif

#ifdef BALANCE_COSTS
   Number cost = 0;
   if (parser.getOption("cost", &cost)) {
      if (cost <= 0) {
         std::cerr << "Bad cost constant = " << cost << "\n";
         exitOnError();
      }
      indexedPriorityQueue.setCostConstant(cost);
   }
#endif

   // Check that we parsed all of the options.
   if (! parser.areOptionsEmpty()) {
      std::cerr << "Error.  Unmatched options:\n";
      parser.printOptions(std::cerr);
      exitOnError();
   }

   setPropensities(std::integral_constant<bool,
                   IndexedPriorityQueue::UsesPropensities>(),
                   &indexedPriorityQueue, &propensities);

   //
   // Time the priority queue.
   //

   ads::Timer timer;
   double elapsedTime;
   // If they did not specify a count.
   if (count == 0) {
      count = 1000;
      do {
         indexedPriorityQueue.clear();
         times = initialTimes;
         for (std::size_t i = 0; i != times.size(); ++i) {
            indexedPriorityQueue.push(i, times[i]);
         }
         timer.tic();
         count *= 2;
         int index;
         for (long i = 0; i != count; ++i) {
            index = indexedPriorityQueue.top();
            times[index] += offsets[index];
            indexedPriorityQueue.pushTop(times[index]);
         }
         elapsedTime = timer.toc();
      }
      while (elapsedTime < 1);
   }
   // They specified a count.
   else {
      indexedPriorityQueue.clear();
      times = initialTimes;
      //std::cerr << "Times = " << times;
      for (std::size_t i = 0; i != times.size(); ++i) {
         indexedPriorityQueue.push(i, times[i]);
      }
      timer.tic();
      int index;
      for (long i = 0; i != count; ++i) {
         index = indexedPriorityQueue.top();
         times[index] += offsets[index];
         indexedPriorityQueue.pushTop(times[index]);
      }
      elapsedTime = timer.toc();
   }

   std::cout << "Meaningless result = " << indexedPriorityQueue.top() << "\n"
             << "Drew " << count << " numbers in " << elapsedTime
             << " seconds.\n"
             << "Cycles per second = " << count / elapsedTime << "\n"
             << "Time per operation = " << elapsedTime / count * 1e9
             << " nanoseconds\n";
   std::cout.setf(std::ios_base::fixed, std::ios_base::floatfield);
   std::cout.precision(0);
   std::cout << elapsedTime / count * 1e9 << "\n";

   return 0;
}

// End of file.
