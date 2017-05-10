// -*- C++ -*-

#if !defined(__performance_numerical_interpolation_InterpolatingFunctionRegularGrid_ipp__)
#error This file is an implementation detail of InterpolatingFunctionRegularGrid.
#endif

#include "stlib/numerical/interpolation/InterpolatingFunctionRegularGrid.h"

#include "stlib/ads/timer/Timer.h"
#include "stlib/ads/utility/ParseOptionsArguments.h"
#include "stlib/container/MultiIndexRangeIterator.h"

#include <algorithm>
#include <iostream>

USING_STLIB_EXT_ARRAY_MATH_OPERATORS;
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
         << programName << " extent0 extent1 ...\n\n";
   exit(1);
}

}

//! The main loop.
int
main(int argc, char* argv[]) {
   typedef numerical::InterpolatingFunctionRegularGrid
      <double, Dimension, Order> F;
   typedef F::Grid Grid;
   typedef F::SizeList SizeList;
   typedef F::Point Point;
   typedef F::BBox BBox;
   typedef container::MultiIndexRangeIterator<Dimension> Iterator;

   ads::ParseOptionsArguments parser(argc, argv);

   // Program name.
   programName = parser.getProgramName();

   // There should be Dimension arguments.
   if(parser.getNumberOfArguments() != Dimension) {
      std::cerr << "Bad arguments.\n"
                << "You gave the arguments:\n";
      parser.printArguments(std::cerr);
      exitOnError();
   }

   SizeList extents;
   for (std::size_t i = 0; i != Dimension; ++i) {
      if(! parser.getArgument(&extents[i])) {
         std::cerr << "Could not parse the argument for the grid extent"\
            " in dimension " << i << ".\n";
         exitOnError();
      }
      if (extents[i] < 2) {
         std::cerr << "Each grid extent must be at least 2.\n";
         exitOnError();
      }
   }

   // There should be no options.
   if(! parser.areOptionsEmpty()) {
      std::cerr << "Error.  Unmatched options:\n";
      parser.printOptions(std::cerr);
      exitOnError();
   }

   // The function values.
   Grid grid(extents, 0.);

   const SizeList cellExtents = extents - std::size_t(1);
   // The positions at which to interpolate.
   std::vector<Point> positions;
   positions.reserve(stlib::ext::product(cellExtents));
   const Iterator end = Iterator::end(cellExtents);
   for (Iterator i = Iterator::begin(cellExtents); i != end; ++i) {
     positions.push_back(ext::convert_array<double>(*i) + 0.5);
   }
   std::random_shuffle(positions.begin(), positions.end());

   // The interpolating functor.
   const Point lower = ext::filled_array<Point>(0);
   const Point upper = ext::convert_array<double>(cellExtents);
   F f(grid, BBox{lower, upper});
   
   // Determine how many times to perform interpolation to consume about one
   // second of run time.
   double meaninglessResult = 0;
   std::size_t count = 1000000 / positions.size() + 1;
#ifdef PERFORMANCE_DERIVATIVE
   Point gradient;
#endif
   ads::Timer timer;
   timer.tic();
   for (std::size_t i = 0; i != count; ++i) {
      for (std::size_t j = 0; j != positions.size(); ++j) {
#ifdef PERFORMANCE_DERIVATIVE
         meaninglessResult += f(positions[j], &gradient);
         meaninglessResult += stlib::ext::sum(gradient);
#else
         meaninglessResult += f(positions[j]);
#endif
      }
   }
   double elapsedTime = timer.toc();
   count = static_cast<std::size_t>(std::max(count / elapsedTime, 1.));

   // Time the interpolation.
   timer.tic();
   for (std::size_t i = 0; i != count; ++i) {
      for (std::size_t j = 0; j != positions.size(); ++j) {
#ifdef PERFORMANCE_DERIVATIVE
         meaninglessResult += f(positions[j], &gradient);
         meaninglessResult += stlib::ext::sum(gradient);
#else
         meaninglessResult += f(positions[j]);
#endif
      }
   }
   elapsedTime = timer.toc();

   std::cout << "Meaningless result = " << meaninglessResult << '\n'
             << "Time per interpolation in nanoseconds:\n"
             << elapsedTime * 1e9 / (count * positions.size()) << '\n';

   return 0;
}
