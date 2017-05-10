// -*- C++ -*-

#ifndef __performance_numerical_random_discrete_main_ipp__
#error This file is an implementation detail.
#endif

#include "stlib/ads/timer/Timer.h"
#include "stlib/ads/utility/ParseOptionsArguments.h"
#include "stlib/container/StaticArrayOfArrays.h"
#include "stlib/ext/vector.h"

#include <algorithm>
#include <iostream>
#include <fstream>

USING_STLIB_EXT_VECTOR_IO_OPERATORS;
using namespace stlib;

int
main(int argc, char* argv[]) {
   ads::ParseOptionsArguments parser(argc, argv);

   // Get the PMF.
   if (parser.getNumberOfArguments() != 1) {
      std::cerr << "Error: You must specify the PMF file.\n";
      exit(1);
   }
   std::vector<double> pmf;
   {
      std::ifstream in(parser.getArgument().c_str());
      in >> pmf;
   }
   std::random_shuffle(pmf.begin(), pmf.end());

   // There should be no more arguments.
   if (! parser.areArgumentsEmpty()) {
      std::cerr << "Error: Un-parsed arguments:\n";
      parser.printArguments(std::cerr);
      exit(1);
   }

   // Construct the random deviate generator.
   Generator::DiscreteUniformGenerator uniform;
   Generator random(&uniform);

#ifdef NUMERICAL_SET_INDEX_BITS
   {
      int indexBits;
      if (parser.getOption("indexBits", &indexBits)) {
         random.setIndexBits(indexBits);
      }
   }
#endif

#ifdef NUMERICAL_USE_INFLUENCE
   container::StaticArrayOfArrays<std::size_t> influence;
   {
      // Independent probabilities.  Each probability only influences itself.
      std::vector<std::size_t> sizes(pmf.size(), 1), values(pmf.size());
      for (std::size_t i = 0; i != values.size(); ++i) {
         values[i] = i;
      }
      influence.rebuild(sizes.begin(), sizes.end(),
                        values.begin(), values.end());
   }
   random.setInfluence(&influence);
#endif
   random.initialize(pmf.begin(), pmf.end());

   // There should be no options.
   if (! parser.areOptionsEmpty()) {
      std::cerr << "Error: Unmatched options:\n";
      parser.printOptions(std::cerr);
      exit(1);
   }

   // Time the generator.
   ads::Timer timer;
   std::size_t count = 1000;
   double elapsedTime;
   std::size_t result = 0;
   do {
      count *= 2;
      timer.tic();
      for (std::size_t n = 0; n != count; ++n) {
         result = random();
      }
      elapsedTime = timer.toc();
   }
   while (elapsedTime < 1);

   std::cout
         << "Meaningless result = " << result << "\n"
         << "CLOCKS_PER_SEC = " << CLOCKS_PER_SEC << "\n"
         << "Generated " << count << " discrete random numbers in "
         << elapsedTime << " seconds.\n"
         << "Time per random number in nanoseconds:\n";
   std::cout.setf(std::ios_base::fixed, std::ios_base::floatfield);
   std::cout.precision(0);
   std::cout
         << elapsedTime / count * 1e9 << "\n";

   return 0;
}
