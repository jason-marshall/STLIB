// -*- C++ -*-

/*!
  \file centerPointMesh.h
  \brief Create a mesh from the boundary and an center point.
*/

/*!
  \page examples_geom_mesh_centerPointMesh Create a mesh from the boundary and an center point.
*/

#ifndef DIMENSION
#error DIMENSION must be defined to compile this program.
#endif

#include "../iss_io.h"

#include "stlib/geom/mesh/iss/build.h"
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
         << programName << " boundary mesh\n"
         << "  boundary is the input file.\n"
         << "  mesh is the output mesh file.\n";
   exit(1);
}

}


//! The main loop.
int
main(int argc, char* argv[]) {
   typedef geom::IndSimpSet<DIMENSION, DIMENSION> Mesh;
   typedef geom::IndSimpSet < DIMENSION, DIMENSION - 1 > Boundary;

   ads::ParseOptionsArguments parser(argc, argv);

   programName = parser.getProgramName();

   // If they did not specify the input and output mesh.
   if (parser.getNumberOfArguments() != 2) {
      std::cerr << "Bad number of required arguments.\n"
                << "You gave the arguments:\n";
      parser.printArguments(std::cerr);
      exitOnError();
   }

   // Read the input boundary.
   Boundary boundary;
   readAscii(parser.getArgument().c_str(), &boundary);

   // Make the mesh.
   Mesh mesh;
   geom::centerPointMesh(boundary, &mesh);

   // Write the mesh file.
   writeAscii(parser.getArgument().c_str(), mesh);

   return 0;
}
