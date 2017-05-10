// -*- C++ -*-

/*!
  \file boundary.h
  \brief Extract the boundary of a simplicial mesh.
*/

/*!
  \page examples_geom_mesh_boundary Extract the boundary of a simplicial mesh


  \section examples_geom_mesh_boundary_example2 2-D Example

  We start with a 2-D mesh in 2-D space.
  \verbatim
  cp ../../../data/geom/mesh/22/a_23.txt mesh.txt
  \endverbatim

  \image html boundary_2_mesh.jpg "A 2-D mesh in 2-D space."
  \image latex boundary_2_mesh.pdf "A 2-D mesh in 2-D space."

  The boundary is a 1-D mesh in 2-D space.  We extract the boundary.

  \verbatim
  boundary22.exe mesh.txt boundary.txt
  \endverbatim

  \image html boundary_2_boundary.jpg "A 1-D mesh in 2-D space."
  \image latex boundary_2_boundary.pdf "A 1-D mesh in 2-D space."
*/

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
         << programName << " mesh boundary\n"
         << "  mesh is the input simplicial mesh file.\n"
         << "  boundary is the output file.\n";
   exit(1);
}

}


//! The main loop.
int
main(int argc, char* argv[]) {
   typedef geom::IndSimpSetIncAdj<SpaceDimension, SimplexDimension> Mesh;
   typedef geom::IndSimpSet < SpaceDimension, SimplexDimension - 1 > Boundary;

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
   Mesh mesh;
   readAscii(parser.getArgument().c_str(), &mesh);

   // Extract the boundary.
   Boundary boundary;
   geom::buildBoundary(mesh, &boundary);

   // Write the boundary file.
   writeAscii(parser.getArgument().c_str(), boundary);

   return 0;
}
