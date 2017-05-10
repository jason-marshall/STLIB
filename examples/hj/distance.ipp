// -*- C++ -*-

/*!
  \file examples/hj/distance.cc
  \brief CONTINUE
*/

/*!
  \page hj_distance Distance

  CONTINUE
*/

#include "stlib/hj/hj.h"

#include "stlib/ads/utility/ParseOptionsArguments.h"
#include "stlib/ads/timer/Timer.h"

#include <iostream>
#include <fstream>
#include <sstream>

#include <cassert>

USING_STLIB_EXT_ARRAY_IO_OPERATORS;
using namespace stlib;

namespace {

// Global variables.

//! The program name.
std::string programName;

// Local functions.

//! Exit with an error message.
void
exitOnError() {
   std::cerr
         << "Bad arguments.  Usage:\n"
         << programName << " [-d dx] [-m max_distance] [-u] input output\n";
   exit(1);
}

}


//! The main loop.
int
main(int argc, char* argv[]) {
   static_assert(Dimension == 2 || Dimension == 3,
                 "The dimension must be 2 or 3.");

   const int N = Dimension;
   typedef container::MultiArray<double, N> MultiArray;
   typedef MultiArray::SizeList SizeList;

   //
   // Parse the program options and arguments.
   //

   ads::ParseOptionsArguments parser(argc, argv);

   programName = parser.getProgramName();

   //
   // Get the command line options.
   //

   // By default, compute signed distance.
   bool signed_distance = ! parser.getOption("u");

   // By default, the dx is 1.
   double dx = 1;
   parser.getOption("d", &dx);
   if (dx <= 0) {
      std::cerr << "Bad value for dx.\n";
      exitOnError();
   }

   // By default, the distance is computed for all points.
   double max_distance = std::numeric_limits<double>::max();
   parser.getOption("m", &max_distance);
   if (max_distance < 0) {
      std::cerr << "Bad value for the maximum distance.\n";
      exitOnError();
   }

   // Check that we parsed all of the options.
   if (! parser.areOptionsEmpty()) {
      std::cerr << "Error.  Unmatched options:\n";
      parser.printOptions(std::cerr);
      exitOnError();
   }

   //
   // Parse the program arguments.
   //

   if (parser.getNumberOfArguments() != 2) {
      std::cerr << "Bad number of required arguments.\n"
                << "You gave the arguments:\n";
      parser.printArguments(std::cerr);
      exitOnError();
   }

   // Read the grid.
   MultiArray grid;
   {
      std::ifstream file(parser.getArgument().c_str(), std::ios_base::binary);
      SizeList extents;
      stlib::ext::read(&extents, file);
      grid.rebuild(extents);
      readElements(&grid, file);
   }

   std::cerr << "Grid size = " << grid.extents() << "\n"
             << "dx = " << dx
             << ", max distance = " << max_distance << "\n";
   if (signed_distance) {
      std::cerr << "Will compute signed distance.\n";
   }
   else {
      std::cerr << "Will compute unsigned distance.\n";
   }


   ads::Timer timer;
   std::cout << "Computing distance...\n";
   timer.tic();

   if (signed_distance) {
      hj::computeSignedDistance(grid, dx, max_distance);
   }
   else {
      hj::computeUnsignedDistance(grid, dx, max_distance);
   }

   double elapsed_time = timer.toc();
   std::cout << "done.\nComputation took " << elapsed_time
             << " seconds.\n";

   // Write the grid.
   {
      std::ofstream file(parser.getArgument().c_str(), std::ios_base::binary);
      stlib::ext::write(grid.extents(), file);
      writeElements(grid, file);
   }

   // There should be no more arguments.
   assert(parser.areArgumentsEmpty());

   return 0;
}
