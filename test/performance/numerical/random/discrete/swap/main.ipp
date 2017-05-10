// -*- C++ -*-

#ifndef __performance_numerical_random_discrete_swap_main_ipp__
#error This file is an implementation detail.
#endif

#include "stlib/ads/timer/Timer.h"
#include "stlib/ads/utility/ParseOptionsArguments.h"
#include "stlib/ext/vector.h"

#include <algorithm>
#include <iostream>
#include <fstream>

namespace std {USING_STLIB_EXT_PAIR_IO_OPERATORS;}
using namespace stlib;

void
exitOnError() {
   std::cerr << "Usage:\n";
   // CONTINUE
   exit(1);
}

int
main(int argc, char* argv[]) {
   ads::ParseOptionsArguments parser(argc, argv);

   // Get the number of events.
   std::size_t size = 0;
   parser.getArgument(&size);

   // Get the number of deviates to draw before resetting the PMF.
   std::size_t numberToDraw = 0;
   parser.getArgument(&numberToDraw);

   // There should be no more arguments.
   if (! parser.areArgumentsEmpty()) {
      std::cerr << "Error: Un-parsed arguments:\n";
      parser.printArguments(std::cerr);
      exitOnError();
   }

   // Construct the random deviate generator.
   Generator::DiscreteUniformGenerator uniform;
   Generator random(&uniform);
   // Initialize it with a PMF that is in descending order. Then the initial
   // sorting has no effect.
   {
      std::vector<double> p(size);
      for (std::size_t i = 0; i != p.size(); ++i) {
         p[i] = p.size() - i;
      }
      random.initialize(p.begin(), p.end());
   }

   // There should be no options.
   if (! parser.areOptionsEmpty()) {
      std::cerr << "Error: Unmatched options:\n";
      parser.printOptions(std::cerr);
      exitOnError();
   }

   // Time the generator.
   ads::Timer timer;
   std::size_t count = 1;
   double elapsedTime;
   double cost;
   do {
      count *= 2;
      elapsedTime = 0;
      cost = 0;
      for (std::size_t n = 0; n != count; ++n) {
         // Set the PMF with ascending values.
         for (std::size_t i = 0; i != random.size(); ++i) {
            random.set(i, random.position(i) + 1);
         }
         timer.tic();
         // Draw the specified number of deviates.
         for (std::size_t i = 0; i != numberToDraw; ++i) {
            random();
         }
         elapsedTime += timer.toc();
         cost += random.cost();
      }
   }
   while (elapsedTime < 1);
   cost /= count;

   random.print(std::cout);
   std::cout
         << "Expected cost for a search = " << cost << '\n'
         << "Meaningless result = " << random() << "\n"
         << "CLOCKS_PER_SEC = " << CLOCKS_PER_SEC << "\n"
         << "Generated " << count* numberToDraw << " discrete random numbers in "
         << elapsedTime << " seconds.\n"
         << "Time per random number in nanoseconds:\n";
   std::cout.setf(std::ios_base::fixed, std::ios_base::floatfield);
   std::cout.precision(0);
   std::cout
         << elapsedTime / (count * numberToDraw) * 1e9 << "\n";

   return 0;
}
