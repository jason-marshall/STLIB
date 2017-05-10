// -*- C++ -*-

/*!
  \file quality.h
  \brief Computes the quality metrics for simplicial meshes.
*/

/*!
  \page examples_geom_mesh_utility_quality Mesh Quality


  \section mesh_quality_introduction Introduction

  This program prints
  information about the quality of a simplicial mesh.


  \section mesh_quality_compiling Compiling.

  The makefile defines the Dimension macro to compile this code
  into \c quality2, which reads triangle meshes in 2-D, and into
  \c quality3, which reads tetrahedral meshes in 3-D.

  \section mesh_quality_usage Usage

  Assess the quality of a triangle mesh with:
  \verbatim
  quality2 triangle_mesh.txt
  \endverbatim
  Assess the quality of a tetrahedron mesh with:
  \verbatim
  quality3 tetrahedron_mesh.txt
  \endverbatim

  The programs read meshes in ascii format.
*/

#include "../iss_io.h"

#include "stlib/geom/mesh/iss/quality.h"
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
         << programName << " input\n"
         << "input is the input mesh.\n";
   exit(1);
}

}


//! The main loop.
int
main(int argc, char* argv[]) {
   typedef geom::IndSimpSet<SpaceDimension, SimplexDimension> ISS;

   ads::ParseOptionsArguments parser(argc, argv);

   programName = parser.getProgramName();

   // If they did not specify the input mesh.
   if (parser.getNumberOfArguments() != 1) {
      std::cerr << "Bad number of required arguments.\n"
                << "You gave the arguments:\n";
      parser.printArguments(std::cerr);
      exitOnError();
   }

   // Read the input mesh.
   ISS mesh;
   readAscii(parser.getArgument().c_str(), &mesh);

   // There should be no more arguments.
   assert(parser.areArgumentsEmpty());

   // Print quality measures for the input mesh.
   geom::printQualityStatistics(std::cout, mesh);

   return 0;
}
