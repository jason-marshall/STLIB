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

using namespace stlib;

//
// Global variables.
//
static std::size_t numberOfPmfToChange;
static double grow, upper;
static container::StaticArrayOfArrays<std::size_t> influence;
static Generator::DiscreteUniformGenerator uniform;
// The random deviate generator.
static Generator discreteRandom(&uniform);

//
// Functions.
//

void
exitOnError() {
   std::cerr << "Usage:\n"
             << "method.exe size numberToChange delta upper\n";
   exit(1);
}

// Do nothing because the discrete generator automatically update the PMF sum.
template<typename _Generator>
inline
void
updateSum(std::true_type /*Automatic update*/, _Generator* /*g*/) {
}

// Tell the discrete generator to update the PMF sum.
template<typename _Generator>
inline
void
updateSum(std::false_type /*Automatic update*/, _Generator* g) {
   g->updateSum();
}

// Update the PMF sum if necessary.
inline
void
updateSum() {
  updateSum(std::integral_constant<bool, Generator::AutomaticUpdate>(),
            &discreteRandom);
}

inline
void
draw() {
   // Draw a discrete deviate.
   const std::size_t deviate = discreteRandom();

   //
   // Change some PMF values.
   //
   std::size_t index = 0;
   double value = 0;
   for (std::size_t i = 0; i != numberOfPmfToChange; ++i) {
      index = influence(deviate, i);
      value = discreteRandom[index] * grow;
      if (value > upper) {
         value = 1.;
      }
      discreteRandom.set(index, value);
   }
   updateSum();
}

int
main(int argc, char* argv[]) {
   ads::ParseOptionsArguments parser(argc, argv);

   if (parser.getNumberOfArguments() != 4) {
      std::cerr << "Error: Wrong number of command line arguments.\n";
      exitOnError();
   }

   // Get the size.
   std::size_t size = 0;
   parser.getArgument(&size);
   if (size == 0) {
      std::cerr << "Error: Bad size.\n";
      exitOnError();
   }

   // Get the number of PMF to change each time.
   numberOfPmfToChange = 0;
   parser.getArgument(&numberOfPmfToChange);
   if (numberOfPmfToChange == 0) {
      std::cerr << "Error: Bad value for the number of PMF to change.\n";
      exitOnError();
   }

   // Get the amount by which to change the PMF.
   double delta = 0;
   parser.getArgument(&delta);
   if (delta <= 0) {
      std::cerr << "Error: Bad value for delta.\n";
      exitOnError();
   }
   grow = 1. + delta;

   // Get the upper bound on the probability range.
   upper = 0;
   parser.getArgument(&upper);
   if (upper <= 1) {
      std::cerr << "Error: Bad value for upper.\n";
      exitOnError();
   }

   // There should be no more arguments.
   if (! parser.areArgumentsEmpty()) {
      std::cerr << "Error: Un-parsed arguments:\n";
      parser.printArguments(std::cerr);
      exitOnError();
   }

   std::vector<double> pmf(size, 1.);

#ifdef NUMERICAL_SET_INDEX_BITS
   {
      int indexBits;
      if (parser.getOption("indexBits", &indexBits)) {
         discreteRandom.setIndexBits(indexBits);
      }
   }
#endif

   // Build the influence data structure.
   {
      const std::size_t increment = std::max(std::size_t(1),
                                             pmf.size() / numberOfPmfToChange);
      // Independent probabilities.  Each probability only influences itself.
      std::vector<std::size_t> sizes(pmf.size(), numberOfPmfToChange);
      std::vector<std::size_t> values(pmf.size() * numberOfPmfToChange);
      std::size_t index = 0;
      for (std::size_t i = 0; i != pmf.size(); ++i) {
         for (std::size_t j = 0; j != numberOfPmfToChange; ++j) {
            values[index++] = (i + j * increment) % size;
         }
      }
      influence.rebuild(sizes.begin(), sizes.end(),
                        values.begin(), values.end());
   }

#ifdef NUMERICAL_USE_INFLUENCE
   discreteRandom.setInfluence(&influence);
#endif
   discreteRandom.initialize(pmf.begin(), pmf.end());

#ifdef NUMERICAL_REBUILD
   {
      // The number of steps to take between rebuilds.
      Generator::Counter stepsBetweenRebuilds = 0;
      if (parser.getOption("rebuild", &stepsBetweenRebuilds)) {
         discreteRandom.setStepsBetweenRebuilds(stepsBetweenRebuilds);
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
      for (std::size_t n = 0; n != count; ++n) {
         draw();
      }
      elapsedTime = timer.toc();
   }
   while (elapsedTime < 1);
   const std::size_t result = discreteRandom();

   //discreteRandom.print(std::cout);

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
