// -*- C++ -*-

/*!
  \file getVertices.h
  \brief Get the vertices from an indexed simplex set file.
*/

/*!
  \page examples_geom_mesh_utility_getVertices Get the vertices from an indexed simplex set file.

  \section getVerticesIntroduction Introduction

  This program gets the vertices from an indexed simplex set file.


  \section getVerticesUsage Usage

  \verbatim
  getVerticesNM.exe mesh vertices
  \endverbatim

  - mesh is the input indexed simplex set file.
    See \ref iss_file_io for a description of the file format.
  - vertices is the output vertices file.

  \section getVerticesExample Example
  CONTINUE.
*/

#include "../iss_io.h"

#include "stlib/ads/utility/ParseOptionsArguments.h"

#include <iostream>
#include <fstream>

#include <cassert>

USING_STLIB_EXT_VECTOR_IO_OPERATORS;
namespace std {USING_STLIB_EXT_ARRAY_IO_OPERATORS;}
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
         << programName << " mesh vertices\n"
         << "- mesh is the input indexed simplex set file.\n"
         << "- vertices is the output vertices file.\n";
   exit(1);
}

}


//! The main loop.
int
main(int argc, char* argv[]) {
   typedef geom::IndSimpSet<SpaceDimension, SimplexDimension> ISS;

   ads::ParseOptionsArguments parser(argc, argv);

   programName = parser.getProgramName();

   // There should be no options.
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
   ISS mesh;
   readAscii(parser.getArgument().c_str(), &mesh);

   std::cout << "The mesh has " << mesh.vertices.size()
             << " vertices and " << mesh.indexedSimplices.size()
             << " simplices.\n";

   // Write the vertices.
   std::ofstream out(parser.getArgument().c_str());
   out.precision(std::numeric_limits<double>::digits10);
   out << mesh.vertices;

   // There should be no more arguments.
   assert(parser.areArgumentsEmpty());

   return 0;
}
