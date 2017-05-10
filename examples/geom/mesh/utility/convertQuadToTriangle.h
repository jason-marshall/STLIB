// -*- C++ -*-

/*!
  \file convertQuadToTriangle.h
  \brief Convert a quadrilateral mesh to a triangle mesh.
*/

/*!
  \page examples_geom_mesh_convertQuadToTriangle Quadrilateral to Triangle

  Convert a quadrilateral mesh to a triangle mesh.
*/

#include "../iss_io.h"

#include "stlib/geom/mesh/iss/build.h"
#include "stlib/geom/mesh/quadrilateral/QuadMesh.h"
#include "stlib/geom/mesh/quadrilateral/file_io.h"
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
         << "  in is the input quadrilateral mesh.\n"
         << "  out is the triangle mesh.\n";
   exit(1);
}
}

//! The main loop.
int
main(int argc, char* argv[]) {
   typedef geom::QuadMesh<Dimension> QuadMesh;
   typedef geom::IndSimpSet<Dimension, 2> ISS;

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
   QuadMesh quadMesh;
   std::ifstream in(parser.getArgument().c_str());
   geom::readAscii(in, &quadMesh);

   // Build the triangle mesh.
   ISS mesh;
   geom::buildFromQuadMesh(quadMesh, &mesh);

   std::cout << "The triangle mesh has " << mesh.vertices.size()
             << " vertices and " << mesh.indexedSimplices.size()
             << " simplices.\n";

   // Write the output mesh.
   writeAscii(parser.getArgument().c_str(), mesh);

   // There should be no more arguments.
   assert(parser.areArgumentsEmpty());

   return 0;
}
