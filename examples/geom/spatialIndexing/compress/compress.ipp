// -*- C++ -*-

/*!
  \file examples/geom/spatialIndexing/compress.ipp
  \brief Compress an image.
*/

#ifndef __examples_geom_spatialIndexing_compress_ipp__
#error This is an implementation detail.
#endif

#include "geom/spatialIndexing/OrthtreeMap.h"
#include "geom/kernel/Point.h"

#include "stlib/ads/array/Array.h"
#include "stlib/ads/timer/Timer.h"
#include "stlib/ads/utility/ParseOptionsArguments.h"
#include "stlib/ads/utility/string.h"

#include "numerical/constants.h"

#include <iostream>
#include <functional>
#include <fstream>

#include <cassert>
#include <cmath>

namespace {

//
// Orthtree types.
//

typedef geom::OrthtreeMap < Dimension, MaximumLevel, Number, AutomaticBalancing,
        geom::SplitNull,
        Average,
        ads::GeneratorConstant<bool>,
        Coarsen >
        Orthtree;
typedef Orthtree::Key Key;
typedef Orthtree::Element Element;
typedef Orthtree::iterator iterator;
typedef Orthtree::const_iterator const_iterator;
//! A Cartesian point.
typedef std::tr1::array<Number, Dimension> Point;

//
// Functors.
//

struct SampledFunction {
   Element
   operator()(const Point& x) const {
      const Number y = magnitude(x);
      return std::exp(- 2 * y) *
             std::cos(4 * numerical::Constants<Number>::Pi() * y);
   }
};

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
         << programName << " variation [original compressed]" << "\n";
   exit(1);
}

}


//! The main loop.
int
main(int argc, char* argv[]) {
   ads::ParseOptionsArguments parser(argc, argv);

   // Program name.
   programName = parser.getProgramName();

   if (!(parser.getNumberOfArguments() == 1 ||
         parser.getNumberOfArguments() == 3)) {
      std::cerr << "Bad number of required arguments.\n"
                << "You gave the arguments:\n";
      parser.printArguments(std::cerr);
      exitOnError();
   }

   Number variation = 0;
   parser.getArgument(&variation);

   std::string originalName;
   if (parser.getNumberOfArguments() != 0) {
      originalName = parser.getArgument();
   }

   std::string compressedName;
   if (parser.getNumberOfArguments() != 0) {
      compressedName = parser.getArgument();
   }

   // There should be no more arguments.
   assert(parser.areArgumentsEmpty());

   // There should be no options.
   if (! parser.areOptionsEmpty()) {
      std::cerr << "Error.  Unmatched options:\n";
      parser.printOptions(std::cerr);
      exitOnError();
   }

   // Construct the orthtree.
   Orthtree orthtree(ext::make_array(-1.), ext::make_array(2.));
   // Set the maximum allowed variation.
   orthtree.getCoarsen().set(variation);
   // Insert a single leaf.
   orthtree.insert(Key());
   // Refine to the maximum level.
   orthtree.refine(ads::constructGeneratorConstant(true));
   // Evaluate the function.
   {
      SampledFunction f;
      orthtree.apply(f);
   }

   std::cout << "The orthtree has " << orthtree.size() << " leaves.\n"
             << "Writing the original orthtree...\n";
   if (! originalName.empty()) {
      std::ofstream out(originalName.c_str());
      geom::printVtkUnstructuredGrid(out, orthtree);
   }
   std::cout << "Done.\n"
             << "Compressing...\n";

   ads::Timer timer;
   timer.tic();

   int countCoarsen = orthtree.coarsen();

   double elapsedTime = timer.toc();

   std::cout << "Done.  Time = " << elapsedTime << " seconds.\n"
             << "The orthtree has " << orthtree.size() << " leaves.\n"
             << "Performed " << countCoarsen << " coarsening operations.\n";

   std::cout << "Writing the compressed orthtree...\n";
   if (! compressedName.empty()) {
      std::ofstream out(compressedName.c_str());
      geom::printVtkUnstructuredGrid(out, orthtree);
   }
   std::cout << "Done.\n";

   return 0;
}
