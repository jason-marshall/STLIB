// -*- C++ -*-

/*!
  \file laplacian21.cc
  \brief Apply sweeps of Laplacian smoothing to the vertices of a curve.
*/

/*!
  \page examples_geom_mesh_laplacian21 Sweeps of Laplacian smoothing on the vertices of a curve.


  \section mesh_laplacian21_introduction Introduction

  This program reads the mesh of a curve, applies sweeps of Laplacian
  smoothing to the vertices and then writes the smoothed mesh.


  \section mesh_laplacian21_usage Usage

  Command line options:
  \verbatim
  laplacian21 [-angle maxAngleDeviation] [-sweeps=n] in out
  \endverbatim
  - maxAngleDeviation specifies the maximum allowed deviation of the
  interior angle (from pi) for moving a vertex.
  - sweeps is used to specify the number of sweeps.
  - in is the file name for the input mesh.
  - out is the file name for the smoothed mesh.

  The programs read and write meshes in ascii format.
*/

#include "../iss_io.h"

#include "stlib/geom/mesh/iss/laplacian.h"
#include "stlib/ads/timer/Timer.h"
#include "stlib/ads/utility/ParseOptionsArguments.h"

#include <string>
#include <sstream>

#include <cassert>


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
         << programName << "\n"
         << "  [-angle=maxAngleDeviation] [-sweeps=n] in out\n"
         << "-angle specifies the maximum allowed deviation of the interior angle\n"
         << "   from pi for moving a vertex.\n"
         << "-sweeps is used to specify the number of sweeps.\n"
         << "in is the file name for the input mesh.\n"
         << "out is the file name for the smoothed mesh.\n";
   exit(1);
}

}


//! The main loop.
int
main(int argc, char* argv[]) {
   typedef geom::IndSimpSetIncAdj<2, 1> ISS;
   typedef geom::PointsOnManifold<2, 1, 1> Manifold;

   ads::ParseOptionsArguments parser(argc, argv);

   programName = parser.getProgramName();

   //
   // Get the options.
   //

   double maxAngleDeviation = -1;
   if (parser.getOption("angle", &maxAngleDeviation)) {
      if (maxAngleDeviation < 0) {
         std::cerr << "Bad value for the maximum angle deviation.\n";
         exitOnError();
      }
   }

   // By default one sweep is performed.
   int numberOfSweeps = 1;
   parser.getOption("sweeps", &numberOfSweeps);
   if (numberOfSweeps < 0) {
      std::cerr << "Bad value for the number of sweeps.\n";
      exitOnError();
   }

   // Check that we parsed all of the options.
   if (! parser.areOptionsEmpty()) {
      std::cerr << "Error.  Unmatched options:\n";
      parser.printOptions(std::cerr);
      exitOnError();
   }

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
   /*
   std::cout << "Quality of the input mesh:\n";
   geom::printQualityStatistics( std::cout, mesh );
   */

   // The manifold description of the curve.
   Manifold manifold(mesh, maxAngleDeviation);
   manifold.insertAtVertices();

   std::cout << "\nSmoothing the boundary vertices..." << std::flush;
   ads::Timer timer;
   double elapsedTime;
   timer.tic();

   // Smooth the mesh.
   geom::applyLaplacian(&mesh, &manifold, numberOfSweeps);

   elapsedTime = timer.toc();
   std::cout << "done.  Operatation took " << elapsedTime << " seconds.\n\n";

   // Print quality measures for the output mesh.
   /*
   std::cout << "Quality of the smoothed mesh:\n";
   geom::printQualityStatistics( std::cout, mesh );
   */

   // Write the output mesh.
   writeAscii(parser.getArgument().c_str(), mesh);

   // There should be no more arguments.
   assert(parser.areArgumentsEmpty());

   return 0;
}
