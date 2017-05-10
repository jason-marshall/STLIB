// -*- C++ -*-

/*!
  \file boundaryIndices.h
  \brief Extract the indices of the boundary nodes of a simplicial mesh.
*/

/*!
  \page examples_geom_mesh_boundaryIndices Extract the indices of the boundary nodes
*/

#include "../iss_io.h"

#include "stlib/geom/mesh/iss/set.h"
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
         << "  boundaryIndices is the output file.\n";
   exit(1);
}

}


//! The main loop.
int
main(int argc, char* argv[]) {
   typedef geom::IndSimpSetIncAdj<SpaceDimension, SimplexDimension> Mesh;

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

   // Get the boundary vertex indices.
   std::vector<std::size_t> indices;
   geom::determineBoundaryVertices(mesh, std::back_inserter(indices));

   // Write the boundary node indices.
   {
      std::ofstream file(parser.getArgument().c_str());
      file << int(indices.size()) << "\n";
      for (std::vector<std::size_t>::const_iterator i = indices.begin();
            i != indices.end(); ++i) {
         file << *i << "\n";
      }
   }

   return 0;
}
