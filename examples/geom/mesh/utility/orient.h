// -*- C++ -*-

/*!
  \file orient.h
  \brief Try to orient the simplices in the mesh.
*/

/*!
  \page examples_geom_mesh_orient Try to orient the simplices in the mesh.

  This program tries to orient the simplices in the mesh.
*/

#include "../iss_io.h"

#include "stlib/geom/mesh/iss/accessors.h"
#include "stlib/geom/mesh/iss/transform.h"
#include "stlib/ads/utility/ParseOptionsArguments.h"

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
         << programName << " in out\n"
         << "  in is the input indexed simplex set.\n"
         << "  out is the oriented mesh.\n";
   exit(1);
}

}


//! The main loop.
int
main(int argc, char* argv[]) {
   typedef geom::IndSimpSetIncAdj<SpaceDimension, SimplexDimension> ISS;

   ads::ParseOptionsArguments parser(argc, argv);

   programName = parser.getProgramName();

   // If they did not specify the input and output mesh.
   if (parser.getNumberOfArguments() != 2) {
      std::cerr << "Bad number of required arguments.\n"
                << "You gave the arguments:\n";
      parser.printArguments(std::cerr);
      exitOnError();
   }

   // Read the input mesh.
   ISS mesh;
   readAscii(parser.getArgument().c_str(), &mesh);

   std::cout << "The mesh has " << mesh.vertices.size()
             << " vertices and " << mesh.indexedSimplices.size()
             << " simplices.\n";

   if (isOriented(mesh)) {
      std::cout << "The mesh is oriented.\n";
   }
   else {
      std::cout << "The mesh is not oriented.\n";
      std::cout << "Trying to orient the mesh...\n";
      orient(&mesh);
      std::cout << "Done.\n";
      if (isOriented(mesh)) {
         std::cout << "The mesh is now oriented.\n";
      }
      else {
         std::cout << "The mesh cannot be oriented.\n";
      }
   }

   // Write the output mesh.
   writeAscii(parser.getArgument().c_str(), mesh);

   // There should be no more arguments.
   assert(parser.areArgumentsEmpty());

   return 0;
}
