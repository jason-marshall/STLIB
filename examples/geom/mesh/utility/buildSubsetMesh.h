// -*- C++ -*-

/*!
  \file buildSubsetMesh.h
  \brief Build a subset of the input mesh.
*/

/*!
  \page examples_geom_mesh_buildSubsetMesh Build a subset of the input mesh.

  \section buildSubsetMeshUsage Usage

  \verbatim
  buildSubsetMeshNM.exe [-nodes=file] [-cells=file] inputMesh outputMesh
  \endverbatim

  - nodes is used to specify an array of node indices.
  - cells is used to specify an array of cell indices.
  - inputMesh is the input mesh.
  - outputMesh is the output mesh.

  Here N is the space dimension and M is the simplex dimension.
  One must use either the nodes or the cells option, but not both.
*/

#include "../iss_io.h"

#include "stlib/geom/mesh/iss/build.h"
#include "stlib/ads/utility/ParseOptionsArguments.h"

#include <sstream>

#include <cassert>

USING_STLIB_EXT_VECTOR_IO_OPERATORS;
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
         << programName
         << " [-nodes=file] [-cells=file] inputMesh outputMesh\n"
         << "- nodes is used to specify an array of node indices.\n"
         << "- cells is used to specify an array of cell indices.\n"
         << "- inputMesh is the input mesh.\n"
         << "- outputMesh is the output mesh.\n"
         << "One must use either the nodes or the cells option, but not both.\n";
   exit(1);
}

}


//! The main loop.
int
main(int argc, char* argv[]) {
   typedef geom::IndSimpSet<SpaceDimension, SimplexDimension> ISS;

   ads::ParseOptionsArguments parser(argc, argv);

   programName = parser.getProgramName();

   // If they did not specify the input and output files.
   if (parser.getNumberOfArguments() != 2) {
      std::cerr << "Bad number of required arguments.\n"
                << "You gave the arguments:\n";
      parser.printArguments(std::cerr);
      exitOnError();
   }

   // Read the input mesh.
   ISS inputMesh;
   readAscii(parser.getArgument().c_str(), &inputMesh);

   ISS outputMesh;
   std::vector<std::size_t> indices;
   std::string indicesFileName;
   if (parser.getOption("nodes", &indicesFileName)) {
      // Read the indices.
      std::ifstream in(indicesFileName.c_str());
      if (! in) {
         std::cerr << "Error reading the node indices input file.\n";
         exitOnError();
      }
      in >> indices;
      geom::buildFromSubsetVertices(inputMesh, indices.begin(), indices.end(),
                                    &outputMesh);
   }
   else if (parser.getOption("cells", &indicesFileName)) {
      // Read the indices.
      std::ifstream in(indicesFileName.c_str());
      if (! in) {
         std::cerr << "Error reading the cell indices input file.\n";
         exitOnError();
      }
      in >> indices;
      geom::buildFromSubsetSimplices(inputMesh, indices.begin(), indices.end(),
                                     &outputMesh);
   }
   else {
      std::cerr << "Error: You must use either the nodes or cells option.\n";
      exitOnError();
   }

   // Check that we parsed all of the options.
   if (! parser.areOptionsEmpty()) {
      std::cerr << "Error.  Unmatched options:\n";
      parser.printOptions(std::cerr);
      exitOnError();
   }

   // Write the output mesh.
   std::ofstream out(parser.getArgument().c_str());
   if (! out) {
      std::cerr << "Error writing the output mesh file.\n";
      exitOnError();
   }
   writeAscii(out, outputMesh);

   // There should be no more arguments.
   assert(parser.areArgumentsEmpty());

   return 0;
}
