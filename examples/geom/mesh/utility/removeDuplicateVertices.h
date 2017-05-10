// -*- C++ -*-

/*!
  \file removeDuplicateVertices.h
  \brief Remove duplicate vertices.

*/

/*!
  \page examples_geom_mesh_removeDuplicateVertices Remove Duplicate Vertices
*/

#include "../iss_io.h"

#include "stlib/geom/mesh/iss/distinct_points.h"
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
         << programName << " [-distance=min] input output\n"
         << "  Vertices within the specified distance are considered to be duplicates.\n"
         << "  input is the input indexed simplex set.\n"
         << "  output is the output mesh.\n";
   exit(1);
}

}


//! The main loop.
int
main(int argc, char* argv[]) {
   typedef geom::IndSimpSet<SpaceDimension, SimplexDimension> Mesh;

   ads::ParseOptionsArguments parser(argc, argv);

   programName = parser.getProgramName();

   //
   // Get the options.
   //

   double minimumDistance = -1;
   parser.getOption("distance", &minimumDistance);

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

   //
   // Remove the duplicate vertices.
   //

   // If the user did not specify a minimum distance.
   if (minimumDistance < 0) {
      geom::removeDuplicateVertices(&mesh);
   }
   else {
      geom::removeDuplicateVertices(&mesh, minimumDistance);
   }

   std::cout << "The output mesh has " << mesh.vertices.size()
             << " vertices and " << mesh.indexedSimplices.size()
             << " simplices.\n";

   // Write the output mesh.
   writeAscii(parser.getArgument().c_str(), mesh);

   // There should be no more arguments.
   assert(parser.areArgumentsEmpty());

   return 0;
}
