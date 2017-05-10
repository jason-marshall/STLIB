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

void
exitOnError() {
   std::cerr << "Usage:\n";
   // CONTINUE
   exit(1);
}

//! Do nothing because the discrete generator automatically update the PMF sum.
template<class Generator>
inline
void
updateSum(std::true_type /*Automatic update*/, Generator* /*random*/) {
}

//! Tell the discrete generator to update the PMF sum.
template<class Generator>
inline
void
updateSum(std::false_type /*Automatic update*/, Generator* random) {
   random->updateSum();
}

//! Update the PMF sum if necessary.
template<class Generator>
inline
void
updateSum(Generator* random) {
  updateSum(std::integral_constant<bool, Generator::AutomaticUpdate>(), random);
}

template<class Generator>
inline
void
draw(Generator* random, const std::size_t numberOfPmfToChange) {
   const double Unity = 1 + 0.1 * std::numeric_limits<double>::epsilon();

   // Draw a deviate.
   std::size_t deviate = (*random)();

   // Change some PMF values.
   const std::size_t Size = random->size();
   for (std::size_t i = 0; i != numberOfPmfToChange; ++i) {
      random->set(deviate, (*random)[deviate] * Unity);
      deviate = (deviate + 1) % Size;
   }
   updateSum(random);
}

template<class Generator>
inline
void
draw(Generator* random) {
   const double Unity = 1 + 0.1 * std::numeric_limits<double>::epsilon();

   // Draw a deviate.
   std::size_t deviate = (*random)();
   // Change the PMF value.
   random->set(deviate, (*random)[deviate] * Unity);
   updateSum(random);
}

int
main(int argc, char* argv[]) {
   ads::ParseOptionsArguments parser(argc, argv);

   // Get the PMF.
   if (!(parser.getNumberOfArguments() == 1 ||
         parser.getNumberOfArguments() == 2)) {
      std::cerr << "Error: Wrong number of command line arguments.\n";
      exitOnError();
   }
   std::vector<double> pmf;
   {
      std::ifstream in(parser.getArgument().c_str());
      in >> pmf;
   }
   std::random_shuffle(pmf.begin(), pmf.end());

   // Get the number of PMF to change each time.
   std::size_t numberOfPmfToChange = 1;
   if (! parser.areArgumentsEmpty()) {
      parser.getArgument(&numberOfPmfToChange);
      if (numberOfPmfToChange > pmf.size()) {
         std::cerr << "Error: Bad value for the number of PMF to change.\n";
         exitOnError();
      }
   }

   // There should be no more arguments.
   if (! parser.areArgumentsEmpty()) {
      std::cerr << "Error: Un-parsed arguments:\n";
      parser.printArguments(std::cerr);
      exitOnError();
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
      std::vector<std::size_t> sizes(pmf.size(), 1);
      std::vector<std::size_t> values(pmf.size());
      for (std::size_t i = 0; i != values.size(); ++i) {
         values[i] = i;
      }
      influence.rebuild(sizes.begin(), sizes.end(),
                        values.begin(), values.end());
   }
   random.setInfluence(&influence);
#endif
   random.initialize(pmf.begin(), pmf.end());

#ifdef NUMERICAL_REBUILD
   {
      // The number of steps to take between rebuilds.
      Generator::Counter stepsBetweenRebuilds = 0;
      if (parser.getOption("rebuild", &stepsBetweenRebuilds)) {
         random.setStepsBetweenRebuilds(stepsBetweenRebuilds);
      }
   }
#endif

   // There should be no options.
   if (! parser.areOptionsEmpty()) {
      std::cerr << "Error: Unmatched options:\n";
      parser.printOptions(std::cerr);
      exitOnError();
   }

   // Time the generator.
   ads::Timer timer;
   std::size_t count = 1000;
   double elapsedTime;
   do {
      count *= 2;
      timer.tic();
      if (numberOfPmfToChange == 1) {
         for (std::size_t n = 0; n != count; ++n) {
            draw(&random);
         }
      }
      else {
         for (std::size_t n = 0; n != count; ++n) {
            draw(&random, numberOfPmfToChange);
         }
      }
      elapsedTime = timer.toc();
   }
   while (elapsedTime < 1);
   const std::size_t result = random();

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
