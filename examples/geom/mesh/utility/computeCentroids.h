// -*- C++ -*-

/*!
  \file computeCentroids.h
  \brief Compute the centroids of the cells.
*/

/*!
  \page examples_geom_mesh_utility_computeCentroids Compute the centroids of the cells in a mesh.

  \section computeCentroidsIntroduction Introduction

  This reads a mesh from an indexed simplex set file and computes the centroid
  of each cell.

  \section computeCentroidsUsage Usage

  \verbatim
  computeCentroidsNM.exe mesh centroids
  \endverbatim

  - mesh is the input indexed simplex set file.
    See \ref iss_file_io for a description of the file format.
  - centroids is the output centroids file.

  \section computeCentroidsExample Example
  CONTINUE.
*/

#include "../iss_io.h"

#include "stlib/geom/mesh/iss/accessors.h"
#include "stlib/ads/utility/ParseOptionsArguments.h"

#include <iostream>
#include <fstream>

#include <cassert>

USING_STLIB_EXT_ARRAY_IO_OPERATORS;
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
         << programName << " mesh centroids\n"
         << "- mesh is the input indexed simplex set file.\n"
         << "- centroids is the output centroids file.\n";
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

   // Write the centroids.
   std::ofstream out(parser.getArgument().c_str());
   out.precision(std::numeric_limits<double>::digits10);
   out << mesh.indexedSimplices.size() << "\n";
   ISS::Vertex centroid;
   for (std::size_t n = 0; n != mesh.indexedSimplices.size(); ++n) {
      geom::getCentroid(mesh, n, &centroid);
      out << centroid << "\n";
   }

   // There should be no more arguments.
   assert(parser.areArgumentsEmpty());

   return 0;
}
