// -*- C++ -*-

/*!
  \file laplacianSolve.h
  \brief Apply Laplacian smoothing to the interior vertices.
*/

/*!
  \page examples_geom_mesh_laplacianSolve Laplacian Smoothing


  \section mesh_laplacianSolve_introduction Introduction

  This program reads a mesh, applies Laplacian smoothing to the interior
  vertices and then writes the smoothed mesh.


  \section mesh_laplacianSolve_compiling Compiling.

  The makefile defines the DIMENSION macro to compile this code
  into \c laplacian2.exe, which smooths triangle meshes in 2-D, and into
  \c laplacian3.exe, which smooths tetrahedral meshes in 3-D.


  \section mesh_laplacianSolve_usage Usage

  Command line options:
  \verbatim
  laplacianN input output
  \endverbatim
  Here N is either 2 or 3.

  The programs read and write meshes in ascii format.
*/

#ifndef DIMENSION
#error DIMENSION must be defined to compile this program.
#endif

#include "../iss_io.h"

#include "stlib/geom/mesh/iss/solveLaplacian.h"

#include "stlib/ads/utility/ParseOptionsArguments.h"

#include <string>
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
         << "in is the file name for the input mesh.\n"
         << "out is the file name for the smoothed mesh.\n";
   exit(1);
}

}


//! The main loop.
int
main(int argc, char* argv[]) {
   typedef geom::IndSimpSetIncAdj<DIMENSION, DIMENSION> ISS;

   if (argc != 3) {
      std::cerr << "Bad arguments.  Usage:\n"
                << *argv << " input output\n";
      exit(1);
   }

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

   // Print quality measures for the input mesh.
   std::cout << "Quality of the input mesh:\n";
   geom::printQualityStatistics(std::cout, mesh);

   std::cout << "\nSmoothing the mesh..." << std::flush;

   geom::solveLaplacian(&mesh);

   std::cout << "done.\n\n";

   // Print quality measures for the output mesh.
   std::cout << "Quality of the smoothed mesh:\n";
   geom::printQualityStatistics(std::cout, mesh);

   // Write the output mesh.
   writeAscii(parser.getArgument().c_str(), mesh);

   // There should be no more arguments.
   assert(parser.areArgumentsEmpty());

   return 0;
}
