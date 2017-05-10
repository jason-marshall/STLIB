// -*- C++ -*-

/*!
  \file splitComponents.h
  \brief Split the connected components of a mesh.
*/

/*!
  \page examples_geom_mesh_splitComponents Split the connected components of a mesh.

  CONTINUE
*/

#include "../iss_io.h"

#include "stlib/geom/mesh/iss/build.h"
#include "stlib/geom/mesh/iss/set.h"
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
         << programName << " in out\n"
         << "  in is the input simplicial mesh file.\n"
         << "  out is the output base name.\n";
   exit(1);
}

}


//! The main loop.
int
main(int argc, char* argv[]) {
   typedef geom::IndSimpSetIncAdj<SpaceDimension, SimplexDimension> ISSIA;
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
   ISSIA mesh;
   readAscii(parser.getArgument().c_str(), &mesh);

   // REMOVE
   std::cerr << "Separate.\n";
   // Separate the components.
   std::vector<std::size_t> delimiters;
   geom::separateComponents(&mesh, std::back_inserter(delimiters));
   // REMOVE
   std::cerr << "Done.\n";

   // Extract each component.
   std::string baseName = parser.getArgument();
   ISS component;
   const std::size_t size = delimiters.size() - 1;
   for (std::size_t n = 0; n != size; ++n) {
      // REMOVE
      std::cerr << n << "\n";
      // Make the file name.
      std::ostringstream name;
      name << baseName << n << ".txt";
      // Build the component.
      geom::buildFromSubsetSimplices(mesh,
                                     ads::constructIntIterator(delimiters[n]),
                                     ads::constructIntIterator(delimiters[n+1]),
                                     &component);
      // Write the component file.
      writeAscii(name.str().c_str(), component);
   }

   // There should be no more arguments.
   assert(parser.areArgumentsEmpty());

   return 0;
}
