// -*- C++ -*-

/*!
  \file fitBoundary.h
  \brief
*/

/*!
  \page examples_geom_mesh_fitBoundary Fit the boundary of a mesh.
*/

#ifndef DIMENSION
#error DIMENSION must be defined to compile this program.
#endif

#include "../iss_io.h"

#include "stlib/geom/mesh/iss/fit.h"
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
         << programName << "\n"
         << "  [-angle=deviation] [-sweeps=s]\n"
         << "  boundary input output\n\n"
         << "  boundary is the boundary mesh.\n"
         << "  input is the input mesh.\n"
         << "  output is the transformed mesh.\n";
   exit(1);
}

}


//! The main loop.
int
main(int argc, char* argv[]) {
   // CONTINUE
   static_assert(DIMENSION == 2, "The dimension must be 2.");
   /*
   static_assert(DIMENSION == 2 || DIMENSION == 3,
   "The dimension must be 2 or 3.");
   */

   typedef geom::IndSimpSetIncAdj<DIMENSION, DIMENSION> Mesh;
   typedef geom::IndSimpSet < DIMENSION, DIMENSION - 1 > Boundary;
   // The functor for computing signed distance.
   typedef geom::ISS_SignedDistance<Boundary> ISS_SD;

   ads::ParseOptionsArguments parser(argc, argv);

   programName = parser.getProgramName();

   //
   // Get the options.
   //

   // By default the deviation angle is 0.
   double deviationAngle = 0;
   parser.getOption("angle", &deviationAngle);
   if (deviationAngle < 0) {
      std::cerr << "Bad deviation angle.\n";
      exitOnError();
   }

   // By default, do one sweep.
   int numberOfSweeps = 1;
   parser.getOption("sweeps", &numberOfSweeps);
   if (numberOfSweeps < 1) {
      std::cerr << "Bad number of sweeps.\n";
      exitOnError();
   }

   // Check that we parsed all of the options.
   if (! parser.areOptionsEmpty()) {
      std::cerr << "Error.  Unmatched options:\n";
      parser.printOptions(std::cerr);
      exitOnError();
   }

   // If they did not specify the boundary, the input, and the output mesh.
   if (parser.getNumberOfArguments() != 3) {
      std::cerr << "Bad number of required arguments.\n"
                << "You gave the arguments:\n";
      parser.printArguments(std::cerr);
      exitOnError();
   }

   //
   // Read the boundary mesh.
   //
   Boundary boundary;
   readAscii(parser.getArgument().c_str(), &boundary);
   // The data structure and functor that computes the signed distance.
   ISS_SD signedDistance(boundary);

   std::cout << "The boundary mesh has " << boundary.vertices.size()
             << " vertices and " << boundary.indexedSimplices.size()
             << " simplices.\n";

   //
   // Read the input mesh.
   //
   Mesh mesh;
   readAscii(parser.getArgument().c_str(), &mesh);

   std::cout << "The input mesh has " << mesh.vertices.size()
             << " vertices and " << mesh.indexedSimplices.size()
             << " simplices.\n";


   std::cout << "Fitting the boundary...\n" << std::flush;
   ads::Timer timer;
   timer.tic();

   // Fit the boundary.
   const double deviationTangent = std::tan(deviationAngle);
   geom::fit(&mesh, signedDistance, deviationTangent, numberOfSweeps);

   double elapsedTime = timer.toc();
   std::cout << "done.\nOperation took " << elapsedTime
             << " seconds.\n";


   // Write the output mesh.
   writeAscii(parser.getArgument().c_str(), mesh);

   // There should be no more arguments.
   assert(parser.areArgumentsEmpty());

   return 0;
}
