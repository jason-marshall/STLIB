// -*- C++ -*-

#ifndef __main_ipp__
#error This file is an implementation detail.
#endif

#include "stlib/numerical/random/discrete/linearSearch.h"

#include "stlib/ads/timer/Timer.h"
#include "stlib/ads/utility/ParseOptionsArguments.h"

#include <algorithm>
#include <iostream>
#include <vector>

using namespace stlib;

int
main(int argc, char* argv[]) {
   ads::ParseOptionsArguments parser(argc, argv);

   if (parser.getNumberOfArguments() != 1) {
      std::cerr << "Error: You must specify the size.\n";
      exit(1);
   }

   // Get the size.
   std::size_t size = 0;
   parser.getArgument(&size);
   if (size <= 0) {
      std::cerr << "Error: bad size " << size << ".\n";
      exit(1);
   }

   std::vector<double> pmf(size, 1);
   std::vector<double> queries(size);
   for (std::size_t i = 0; i != queries.size(); ++i) {
      queries[i] = i;
   }
   std::random_shuffle(queries.begin(), queries.end());

   // There should be no more arguments.
   if (! parser.areArgumentsEmpty()) {
      std::cerr << "Error: Un-parsed arguments:\n";
      parser.printArguments(std::cerr);
      exit(1);
   }

   // There should be no options.
   if (! parser.areOptionsEmpty()) {
      std::cerr << "Error: Unmatched options:\n";
      parser.printOptions(std::cerr);
      exit(1);
   }

   ads::Timer timer;
   std::size_t count = 1000;
   double elapsedTime;
   std::size_t result = 0;
   do {
      count *= 2;
      timer.tic();
      for (std::size_t n = 0; n != count; ++n) {
         for (std::size_t i = 0; i != queries.size(); ++i) {
            result += numerical::SEARCH(pmf.begin(), pmf.end(), queries[i]);
         }
      }
      elapsedTime = timer.toc();
   }
   while (elapsedTime < 1);

   std::cout
         << "Meaningless result = " << result << "\n"
         << "CLOCKS_PER_SEC = " << CLOCKS_PER_SEC << "\n"
         << "Performed " << (count * size) << " searches in "
         << elapsedTime << " seconds.\n"
         << "Time per search in nanoseconds:\n";
   std::cout.setf(std::ios_base::fixed, std::ios_base::floatfield);
   std::cout.precision(0);
   std::cout
         << elapsedTime / (count * size) * 1e9 << "\n";

   return 0;
}
