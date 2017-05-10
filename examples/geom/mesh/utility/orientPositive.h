// -*- C++ -*-

/*!
  \file orientPositive.h
  \brief Orients simplicial meshes.
*/

/*!
  \page examples_geom_mesh_orientPositive Orienting Simplicial Meshes


  \section mesh_orientPositive_introduction Introduction

  This program reads
  a mesh, orients the simplices so that they have non-negative content
  and then writes the oriented mesh.


  \section driver_compiling Compiling.

  The makefile defines the Dimension macro to compile this code
  into \c orientPositive2, which orients triangle meshes in 2-D, and into
  \c orientPositive3, which orients tetrahedral meshes in 3-D.


  \section examples_geom_mesh_orientPositive_usage Usage

  Orient a triangle mesh with:
  \verbatim
  orientPositive2 mesh.txt oriented_mesh.txt
  \endverbatim
  Orient a tetrahedral mesh with:
  \verbatim
  orientPositive3 mesh.txt oriented_mesh.txt
  \endverbatim

  The programs read and write meshes in ascii format.
*/

#include "../iss_io.h"

#include "stlib/geom/mesh/iss/transform.h"
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
         << programName << " input output\n"
         << "input is the input mesh.\n"
         << "output is the output mesh wish positively oriented simplices.\n";
   exit(1);
}

}


//! The main loop.
int
main(int argc, char* argv[]) {
   typedef geom::IndSimpSet<Dimension, Dimension> ISS;

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
   ISS mesh;
   readAscii(parser.getArgument().c_str(), &mesh);

   // Orient the mesh.
   orientPositive(&mesh);

   // Print quality measures for the oriented mesh.
   geom::printQualityStatistics(std::cout, mesh);

   // Write the output mesh.
   writeAscii(parser.getArgument().c_str(), mesh);

   // There should be no more arguments.
   assert(parser.areArgumentsEmpty());

   return 0;
}
