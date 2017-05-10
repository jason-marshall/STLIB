// -*- C++ -*-

/*!
  \file allSurfaces.cc
  \brief Driver for computing a triangulation of a protein surface.
*/

/*!
\page mst_allSurfaces Generate all the surfaces of a molecule.

\section mst_allSurfacesIntroduction Introduction

This driver reads a molecule in xyzr format, triangulates all the
surfaces (not just the visible surface), and then write the triangle
surface mesh as an indexed simplex set.  This driver is only useful
for testing purposes.

\section mst_driverUsage Usage

\verbatim
allSurfaces.exe [-unindexed] [-length=l] [-radius=r] input.xyzr output.txt
\endverbatim

- The unindexed option specifies that the triangles will be generated and
  output in un-indexed format.
- length specifies the maximum allowed edge length.  By default it is 1.
- radius specifies the probe radius.  By default it is 0.
- input.xyzr is the input file containing the atom's centers and radii.
- output.txt will contain the surface triangulation.  See
  \ref iss_file_io for a description of the file format.
*/

#include "stlib/mst/readXyzr.h"
#include "stlib/mst/tesselate_sphere.h"
#include "stlib/mst/tesselate_sphere_unindexed.h"

#include "stlib/ads/utility/ParseOptionsArguments.h"
#include "stlib/geom/mesh/iss/file_io.h"

#include <vector>
#include <sstream>
#include <string>

#include <cassert>

USING_STLIB_EXT_ARRAY_IO_OPERATORS;
using namespace stlib;

namespace {

//
// Global variables.
//

//! The program name.
std::string programName;

//
// Local functions.
//

//! Exit with an error message.
void
exitOnError() {
   std::cerr
         << "Bad arguments.  Usage:\n"
         << programName << " [unindexed] [-length=l] [-radius=r] input.xyzr output.txt\n"
         << "- The unindexed option specifies that the triangles will be generated and\n"
         << "  output in un-indexed format.\n"
         << "- length specifies the maximum allowed edge length.  By default it is 1.\n"
         << "- radius specifies the probe radius.  By default it is 0.\n"
         << "  input.xyzr is the input file containing the atom's centers and radii.\n"
         << "  output.txt will contain the surface triangulation.\n";
   exit(1);
}

}


//! The main loop.
int
main(int argc, char* argv[]) {
   typedef double NumberType;
   typedef std::array<NumberType, 3> Point;

   ads::ParseOptionsArguments parser(argc, argv);

   // Program name.
   programName = parser.getProgramName();

   // If they did not specify the input and output files.
   if (parser.getNumberOfArguments() != 2) {
      std::cerr << "Bad number of required arguments.\n"
                << "You gave the arguments:\n";
      parser.printArguments(std::cerr);
      exitOnError();
   }

   // Maximum edge length.
   NumberType maximumEdgeLength = 1;
   parser.getOption("length", &maximumEdgeLength);
   if (maximumEdgeLength <= 0) {
      std::cerr << "Bad value for the maximum edge length.\n";
      exitOnError();
   }

   // Probe radius.
   NumberType probeRadius = 0;
   parser.getOption("radius", &probeRadius);

   // The atom centers and radii.
   std::vector<Point> centers;
   std::vector<NumberType> radii;

   // Read the input file.
   std::cout << "Reading the input file...\n" << std::flush;
   mst::readXyzr<NumberType>(parser.getArgument().c_str(),
                             std::back_inserter(centers),
                             std::back_inserter(radii));

   // Offset by the probe radius.
   if (probeRadius != 0) {
      for (std::vector<NumberType>::iterator i = radii.begin(); i != radii.end();
            ++i) {
         // Make sure we don't have an atom with negative radius.
         if (*i + probeRadius <= 0) {
            std::cerr << "Bad value for the probe radius.\n";
            exitOnError();
         }
         *i = *i + probeRadius;
      }
   }

   std::cout << "The protein has " << int(centers.size()) << " atoms.\n";

   if (parser.getOption("unindexed")) {
      // Triangulation of the surface.
      std::vector<Point> surfaceTriangulation;

      // Tesselate all of the spheres.
      std::cout << "Tesselating the surface...\n" << std::flush;
      mst::tesselateAllSurfacesUnindexed
      (centers.begin(), centers.end(),
       radii.begin(), radii.end(),
       maximumEdgeLength,
       std::back_inserter(surfaceTriangulation));

      std::cout << "The surface tesselation has "
                << int(surfaceTriangulation.size()) / 3
                << " triangles.\n" << std::flush;

      // Write the tesselation of the surface.
      std::cout << "Writing the surface file...\n" << std::flush;
      {
         std::ofstream out(parser.getArgument().c_str());
         const std::size_t Size = surfaceTriangulation.size();
         // Triples of points form triangles.
         assert(Size % 3 == 0);
         // Write the number of triangles.
         out << Size / 3 << "\n";
         // For each point of a triangle.
         for (std::size_t i = 0; i != Size; ++i) {
            // Write the point.
            out << surfaceTriangulation[i] << "\n";
         }
      }
      std::cout << "Done.\n" << std::flush;
   }
   else {
      // Triangulation of the surface.
      geom::IndSimpSetIncAdj<3, 2, NumberType> surfaceTriangulation;

      // Tesselate all of the spheres.
      std::cout << "Tesselating the surface...\n" << std::flush;
      mst::tesselateAllSurfaces(centers.begin(), centers.end(),
                                radii.begin(), radii.end(),
                                maximumEdgeLength,
                                &surfaceTriangulation);

      std::cout << "The surface tesselation has "
                << surfaceTriangulation.vertices.size()
                << " vertices and "
                << surfaceTriangulation.indexedSimplices.size()
                << " triangles.\n" << std::flush;

      // Write the tesselation of the surface.
      std::cout << "Writing the surface file...\n" << std::flush;
      {
         std::ofstream out(parser.getArgument().c_str());
         geom::writeAscii(out, surfaceTriangulation);
      }
      std::cout << "Done.\n" << std::flush;
   }

   // Check that we parsed all of the options.
   if (! parser.areOptionsEmpty()) {
      std::cerr << "Error.  Unmatched options:\n";
      parser.printOptions(std::cerr);
      exitOnError();
   }

   // There should be no more arguments.
   assert(parser.areArgumentsEmpty());

   return 0;
}
