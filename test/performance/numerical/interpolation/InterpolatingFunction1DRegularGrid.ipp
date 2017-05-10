// -*- C++ -*-

#if !defined(__performance_numerical_interpolation_InterpolatingFunction1DRegularGrid_ipp__)
#error This file is an implementation detail of InterpolatingFunction1DRegularGrid.
#endif

#include "stlib/numerical/interpolation/InterpolatingFunction1DRegularGrid.h"

#include "stlib/ads/timer/Timer.h"
#include "stlib/ads/utility/ParseOptionsArguments.h"

#include <algorithm>
#include <iostream>
#include <vector>

using namespace stlib;

namespace {

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
         << programName << " size\n\n";
   exit(1);
}

}

//! The main loop.
int
main(int argc, char* argv[]) {
   ads::ParseOptionsArguments parser(argc, argv);

   // Program name.
   programName = parser.getProgramName();

   // There should be one argument.
   if(parser.getNumberOfArguments() != 1) {
      std::cerr << "Bad arguments.\n"
                << "You gave the arguments:\n";
      parser.printArguments(std::cerr);
      exitOnError();
   }

   std::size_t size;
   if(! parser.getArgument(&size)) {
      std::cerr << "Could not parse the argument for the grid size.\n";
      exitOnError();
   }
   if (size < 2) {
      std::cerr << "The grid size must be at least 2.\n";
      exitOnError();
   }

   // There should be no more options.
   if(! parser.areOptionsEmpty()) {
      std::cerr << "Error.  Unmatched options:\n";
      parser.printOptions(std::cerr);
      exitOnError();
   }

   // The function values.
   std::vector<double> functionValues(size);
   for (std::size_t i = 0; i != functionValues.size(); ++i) {
      functionValues[i] = i;
   }

   // The positions at which to interpolate.
   const std::size_t numberOfCells = size - 1;
   std::vector<double> x(numberOfCells);
   for (std::size_t i = 0; i != x.size(); ++i) {
      x[i] = i;
   }
   std::random_shuffle(x.begin(), x.end());

   // The interpolating functor.
   numerical::InterpolatingFunction1DRegularGrid<double, Order> 
         f(functionValues.begin(), functionValues.end(), 0., numberOfCells);
   
   // Determine how many times to perform interpolation to consume about one
   // second of run time.
   double meaninglessResult = 0;
   std::size_t count = 1000000 / x.size() + 1;
#ifdef PERFORMANCE_DERIVATIVE
   double derivative;
#endif
   ads::Timer timer;
   timer.tic();
   for (std::size_t i = 0; i != count; ++i) {
      for (std::size_t j = 0; j != x.size(); ++j) {
#ifdef PERFORMANCE_DERIVATIVE
         meaninglessResult += f(x[j], &derivative);
         meaninglessResult += derivative;
#else
         meaninglessResult += f(x[j]);
#endif
      }
   }
   double elapsedTime = timer.toc();
   count = static_cast<std::size_t>(std::max(count / elapsedTime, 1.));

   // Time the interpolation.
   timer.tic();
   for (std::size_t i = 0; i != count; ++i) {
      for (std::size_t j = 0; j != x.size(); ++j) {
#ifdef PERFORMANCE_DERIVATIVE
         meaninglessResult += f(x[j], &derivative);
         meaninglessResult += derivative;
#else
         meaninglessResult += f(x[j]);
#endif
      }
   }
   elapsedTime = timer.toc();

   std::cout << "Meaningless result = " << meaninglessResult << '\n'
             << "Time per interpolation in nanoseconds:\n"
             << elapsedTime * 1e9 / (count * x.size()) << '\n';

   return 0;
}
