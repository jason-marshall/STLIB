// -*- C++ -*-

/*!
  \file identifyLowQualityWithCondNum.h
  \brief Identify the low quality elements.
*/

/*!
  \page examples_geom_mesh_utility_identifyLowQualityWithCondNum Identify the Low Quality Elements


  \section mesh_identifyLowQualityWithCondNum_introduction Introduction

  This program identifies the low quality elements in a mesh.

  \section mesh_identifyLowQualityWithCondNum_compiling Compiling.

  The makefile defines the SpaceDimension and SimplexDimension macros to
  compile this code the various outputs.

  \section mesh_identifyLowQualityWithCondNum_usage Usage

  Identify the low quality elements of a tetrahedron mesh with:
  \verbatim
  identifyLowQualityWithCondNum33.exe minQuality mesh.txt indices.txt
  \endverbatim

  On a MacBook Pro with a 2.4 GHz Intel Core 2 Duo processor, measuring the
  quality takes about 250 nanoseconds per mesh element.
*/

#include "../iss_io.h"

#include "stlib/geom/mesh/iss/quality.h"
#include "stlib/ads/timer/Timer.h"
#include "stlib/ads/utility/ParseOptionsArguments.h"

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
         << programName << " minQuality mesh.txt indices.txt\n"
         << "minQuality is a number between 0 and 1 that specifies the minimum\n"
         << "  allowed quality.\n"
         << "mesh.txt defines the input mesh.\n"
         << "indices.txt is filled with the indices of the low quality elements.\n";
   exit(1);
}

}


//! The main loop.
int
main(int argc, char* argv[]) {
   typedef geom::IndSimpSet<SpaceDimension, SimplexDimension> Mesh;

   ads::ParseOptionsArguments parser(argc, argv);

   programName = parser.getProgramName();

   // If they did not specify the quality, input mesh, and output file.
   if (parser.getNumberOfArguments() != 3) {
      std::cerr << "Bad number of required arguments.\n"
                << "You gave the arguments:\n";
      parser.printArguments(std::cerr);
      exitOnError();
   }

   // The minimum allowed quality.
   double minQuality;
   parser.getArgument(&minQuality);
   if (!(0 < minQuality && minQuality <= 1)) {
      std::cerr << "Bad value for the minimum allowed quality.\n";
      exitOnError();
   }

   // Read the input mesh.
   Mesh mesh;
   readAscii(parser.getArgument().c_str(), &mesh);

   // The output file.
   std::ofstream out(parser.getArgument().c_str());

   // There should be no more arguments.
   assert(parser.areArgumentsEmpty());

   std::vector<std::size_t> indices;

   // Determine the low quality elements.
   ads::Timer timer;
   timer.tic();
   geom::identifyLowQualityWithCondNum(mesh, minQuality,
                                       std::back_inserter(indices));
   double elapsedTime = timer.toc();

   // Write the indices of the bad elements.
   out << indices;

   std::cout << "Checked " << mesh.indexedSimplices.size() << " elements in "
             << elapsedTime << " seconds.\n";
   if (mesh.indexedSimplices.size() != 0) {
      std::cout << "Time per element = "
                << elapsedTime * 1e9 / mesh.indexedSimplices.size()
                << " nanoseconds.\n";
   }

   return 0;
}
