// -*- C++ -*-

/*!
  \file laplacianBoundary.h
  \brief Apply sweeps of Laplacian smoothing to the boundary vertices.
*/

/*!
  \page examples_geom_mesh_laplacianBoundary Sweeps of Laplacian smoothing on boundary vertices.


  \section mesh_laplacianBoundary_introduction Introduction

  This program reads a mesh, applies sweeps of Laplacian smoothing to the
  boundary vertices and then writes the smoothed mesh.


  \section mesh_laplacianBoundary_usage Usage

  Command line options:
  \verbatim
  laplacianBoundaryN [-boundary=file] [-angle=maxDeviation] [-sweeps=n] in out

  \endverbatim
  Here N is either 2 or 3.
  - -boundary specifies a boundary that defines the surface.
  - -angle specifies the maximum allowed deviation of the interior angle
  (from pi in 2-D or 2 * pi in 3-D) for moving a vertex.
  - -sweeps is used to specify the number of sweeps.
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
         << programName << "\n"
         << "  [-boundary=file] [-angle=maxAngleDeviation] [-sweeps=n] in out\n"
         << "-boundary specifies a boundary that defines the surface.\n"
         << "-angle specifies the maximum allowed deviation of the interior angle\n"
         << " (from pi in 2-D or 2 * pi in 3-D) for moving a vertex.\n"
         << "-sweeps is used to specify the number of sweeps.\n"
         << "in is the file name for the input mesh.\n"
         << "out is the file name for the smoothed mesh.\n";
   exit(1);
}

}


//! The main loop.
int
main(int argc, char* argv[]) {
   typedef geom::IndSimpSetIncAdj < Dimension, Dimension - 1 > ISS;
   //typedef geom::IndSimpSetIncAdj<Dimension,Dimension-1> ISS_Boundary;
   typedef geom::IndSimpSet < Dimension, Dimension - 1 > Boundary;
   // The functor for computing signed distance.
   typedef geom::ISS_SignedDistance<Boundary> ISS_SD;
   // The functor for computing the closest point.
   typedef geom::ISS_SD_ClosestPointDirection<Boundary> ISS_SD_CP;

   ads::ParseOptionsArguments parser(argc, argv);

   programName = parser.getProgramName();

   //
   // Get the options.
   //

   // By default one sweep is performed.
   int numberOfSweeps = 1;
   parser.getOption("sweeps", &numberOfSweeps);

   std::string boundaryFileName;
   bool areUsingSeparateBoundary = false;
   if (parser.getOption("boundary", &boundaryFileName)) {
      areUsingSeparateBoundary = true;
   }

   double maxAngleDeviation = 0;
   parser.getOption("angle", &maxAngleDeviation);
   if (maxAngleDeviation < 0) {
      std::cerr << "Bad value for the maximum angle deviation.\n";
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
   geom::printQualityStatistics(std::cout, mesh);
   */

   ads::Timer timer;
   double elapsedTime;

   Boundary boundary;
   if (areUsingSeparateBoundary) {
      // Read the boundary file.
      readAscii(boundaryFileName.c_str(), &boundary);
   }
   else {
      boundary = mesh;
   }

   // The data structure and functor that computes the signed distance and
   // closest point.
   ISS_SD signedDistance(boundary);
   // The functor that returns the closest point.
   ISS_SD_CP closestPoint(signedDistance);

   std::cout << "\nSmoothing the boundary vertices..." << std::flush;
   timer.tic();

   // Smooth the mesh.
   geom::applyLaplacian(&mesh, closestPoint, maxAngleDeviation, numberOfSweeps);

   elapsedTime = timer.toc();
   std::cout << "done.  Operatation took " << elapsedTime << " seconds.\n\n";

   // Print quality measures for the output mesh.
   /*
   std::cout << "Quality of the smoothed mesh:\n";
   geom::printQualityStatistics(std::cout, mesh);
   */

   // Write the output mesh.
   writeAscii(parser.getArgument().c_str(), mesh);

   // There should be no more arguments.
   assert(parser.areArgumentsEmpty());

   return 0;
}
