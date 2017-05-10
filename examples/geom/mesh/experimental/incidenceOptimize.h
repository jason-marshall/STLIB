// -*- C++ -*-

/*!
  \file incidenceOptimize.h
  \brief Optimize the incidence relations.
*/

/*!
  \page examples_geom_mesh_incidenceOptimize Optimize the incidence relations.
*/

#ifndef SPACE_DIMENSION
#error SPACE_DIMENSION must be defined to compile this program.
#endif
#ifndef SIMPLEX_DIMENSION
#error SIMPLEX_DIMENSION must be defined to compile this program.
#endif

#include "../smr_io.h"

#include "stlib/geom/mesh/simplicial/inc_opt.h"
#include "stlib/geom/mesh/simplicial/quality.h"
#include "stlib/ads/timer/Timer.h"
#include "stlib/ads/utility/ParseOptionsArguments.h"

#include <string>
#include <sstream>

#include <cassert>

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
         << programName << " [-norm=n] input output\n"
         << "norm should by 0, 1, or 2 for max norm, 1-norm, or 2-norm.\n"
         << "input is the input mesh.\n"
         << "output is the output mesh.\n";
   exit(1);
}

}


//! The main loop.
int
main(int argc, char* argv[]) {
   static_assert(SPACE_DIMENSION == 2 || SPACE_DIMENSION == 3,
                 "The space dimension must be 2 or 3.");
   static_assert(SIMPLEX_DIMENSION == 2, "The simplex dimension must be 2.");

   typedef geom::SimpMeshRed<SPACE_DIMENSION, SIMPLEX_DIMENSION> Mesh;

   ads::ParseOptionsArguments parser(argc, argv);

   programName = parser.getProgramName();

   //
   // Get the options.
   //

   int norm = 0;
   parser.getOption("norm", &norm);
   if (norm < 0 || norm > 2) {
      std::cerr << "Bad value for the norm.  You specified " << norm << ".\n";
      exitOnError();
   }

   // Check that we parsed all of the options.
   if (! parser.areOptionsEmpty()) {
      std::cerr << "Error.  Unmatched options:\n";
      parser.printOptions(std::cerr);
      exitOnError();
   }

   // If they did not specify the input and output mesh.
   if (parser.getNumberOfArguments() != 2) {
      std::cerr << "Bad number of required arguments.\n"
                << "You gave the arguments:\n";
      parser.printArguments(std::cerr);
      exitOnError();
   }

   // Read the input mesh.
   Mesh mesh;
   readAscii(parser.getArgument().c_str(), &mesh);

   // Print quality measures for the input mesh.
   std::cout << "Quality of the input mesh:\n";
   geom::printQualityStatistics(std::cout, mesh);

   std::cout << "\nOptimizing the topology...\n" << std::flush;
   ads::Timer timer;
   timer.tic();

   const int count = geom::incidenceOptimize(&mesh, norm);

   double elapsedTime = timer.toc();
   std::cout << "done.\n"
             << "Number of edges flipped = " << count << "\n"
             << "Optimization took " << elapsedTime << " seconds.\n";

   // Print quality measures for the output mesh.
   std::cout << "\nQuality of the output mesh:\n";
   geom::printQualityStatistics(std::cout, mesh);

   // Write the output mesh.
   writeAscii(parser.getArgument().c_str(), mesh);

   // There should be no more arguments.
   assert(parser.areArgumentsEmpty());

   return 0;
}
