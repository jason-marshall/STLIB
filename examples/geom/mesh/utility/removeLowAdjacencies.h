// -*- C++ -*-

/*!
  \file removeLowAdjacencies.h
  \brief Remove simplices until there are none with minimum adjacencies less than specified.

*/

/*!
  \page examples_geom_mesh_removeLowAdjacencies Remove Simplices with Low Adjacencies.

  When meshing objects, one usually wants to avoid boundary simplices with
  low adjacencies (boundary simplices with few neighbors).  Consider a triangle
  mesh in 2-D.  A boundary triangle that has only one neighbor is problematic.
  If it lies on smooth portion of the boundary, the triangle will be nearly
  colinear and thus have poor quality.
*/

#include "../iss_io.h"

#include "stlib/geom/mesh/iss/transform.h"
#include "stlib/ads/utility/ParseOptionsArguments.h"

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
         << programName << " [-min=m] input output\n"
         << "  min specifies the minimum required adjacency count.  By default it is\n"
         << "  the mesh dimension.\n"
         << "  input is the input indexed simplex set.\n"
         << "  output is the output mesh.\n";
   exit(1);
}

}


//! The main loop.
int
main(int argc, char* argv[]) {
   typedef geom::IndSimpSetIncAdj<SpaceDimension, SimplexDimension> Mesh;

   ads::ParseOptionsArguments parser(argc, argv);

   programName = parser.getProgramName();

   //
   // Get the options.
   //

   int minimumAdjacencies = SimplexDimension;
   parser.getOption("min", &minimumAdjacencies);

   // Check that we parsed all of the options.
   if (! parser.areOptionsEmpty()) {
      std::cerr << "Error.  Unmatched options:\n";
      parser.printOptions(std::cerr);
      exitOnError();
   }

   // If they did not specify the input and output files.
   if (parser.getNumberOfArguments() != 2) {
      std::cerr << "Bad number of required arguments.\n"
                << "You gave the arguments:\n";
      parser.printArguments(std::cerr);
      exitOnError();
   }

   // Read the input mesh.
   Mesh mesh;
   readAscii(parser.getArgument().c_str(), &mesh);

   std::cout << "The input mesh has " << mesh.vertices.size()
             << " vertices and " << mesh.indexedSimplices.size()
             << " simplices.\n";

   // Remove the simplices with low adjacencies.
   geom::removeLowAdjacencies(&mesh, minimumAdjacencies);

   std::cout << "The ouptut mesh has " << mesh.vertices.size()
             << " vertices and " << mesh.indexedSimplices.size()
             << " simplices.\n";

   // Write the output mesh.
   writeAscii(parser.getArgument().c_str(), mesh);

   // There should be no more arguments.
   assert(parser.areArgumentsEmpty());

   return 0;
}
