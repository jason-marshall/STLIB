// -*- C++ -*-

/*!
  \file subdivide.h
  \brief Subdivide the mesh by splitting elements.
*/

/*!
  \page examples_geom_mesh_subdivide Subdivide the mesh by splitting elements.

  CONTINUE.
*/

#include "../iss_io.h"

#include "stlib/geom/mesh/iss/quality.h"
#include "stlib/geom/mesh/iss/subdivide.h"
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
         << programName << " [-number=n] in out\n"
         << "- number specifies the number of subdivisions.\n"
         << "- in is the input mesh.\n"
         << "- out is the output mesh.\n";
   exit(1);
}

}


//! The main loop.
int
main(int argc, char* argv[]) {
  static_assert((SpaceDimension == 2 && SimplexDimension == 1) ||
                (SpaceDimension == 3 && SimplexDimension == 1) ||
                (SpaceDimension == 2 && SimplexDimension == 2) ||
                (SpaceDimension == 3 && SimplexDimension == 2),
                "Those dimensions are not supported.");

   typedef geom::IndSimpSetIncAdj<SpaceDimension, SimplexDimension> Mesh;

   ads::ParseOptionsArguments parser(argc, argv);

   programName = parser.getProgramName();

   //
   // Get the options.
   //

   // The number of subdivisions.
   int numSubdivisions = 1;
   if (parser.getOption("number", &numSubdivisions)) {
      if (numSubdivisions < 0) {
         std::cerr << "Bad value for the number of subdivisions.\n";
         exitOnError();
      }
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

   // CONTINUE
#ifndef NO_QUALITY_STATISTICS
   // Print quality measures for the input mesh.
   std::cout << "Quality of the input mesh:\n";
   geom::printQualityStatistics(std::cout, mesh);
#endif

   std::cout << "Will perform " << numSubdivisions << " subdivisions.\n"
             << "Subdividing the mesh...\n" << std::flush;
   ads::Timer timer;
   timer.tic();

   // Subdivide the mesh.
   while (numSubdivisions-- >= 1) {
      Mesh subdividedMesh;
      geom::subdivide(mesh, &subdividedMesh);
      mesh.swap(subdividedMesh);
   }

   double elapsedTime = timer.toc();
   std::cout << "done.\n"
             << "Subdivision took " << elapsedTime << " seconds.\n";

   // CONTINUE
#ifndef NO_QUALITY_STATISTICS
   // Print quality measures for the subdivided mesh.
   std::cout << "\nQuality of the subdivided mesh:\n";
   geom::printQualityStatistics(std::cout, mesh);
#endif

   // Write the output mesh.
   writeAscii(parser.getArgument().c_str(), mesh);

   // There should be no more arguments.
   assert(parser.areArgumentsEmpty());

   return 0;
}
