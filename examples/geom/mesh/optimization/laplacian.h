// -*- C++ -*-

/*!
  \file laplacian.h
  \brief Apply sweeps of Laplacian smoothing to the interior vertices.
*/

/*!
  \page examples_geom_mesh_laplacian Sweeps of Laplacian Smoothing


  \section mesh_laplacian_introduction Introduction

  This program reads a mesh, applies sweeps of Laplacian smoothing to the
  interior vertices and then writes the smoothed mesh.


  \section mesh_laplacian_usage Usage

  Command line options:
  \verbatim
  laplacianN [-closestPoint] [-boundary=file] [-onlyBoundary=file]
             [-angle=maxDeviation] [-sweeps=n] in out
  \endverbatim
  Here N is either 2 or 3.
  - -closestPoint specifies that the closest point (instead of the closest
    point in the normal direction) should be used.
  - -boundary specifies that the boundary should be smoothed as well.
  - -onlyBoundary specifies that only the boundary should be smoothed.
  - -angle specifies the maximum allowed deviation of the interior angle
  (from pi in 2-D or 2 * pi in 3-D) for moving a vertex.
  - -sweeps is used to specify the number of sweeps.
  - in is the file name for the input mesh.
  - out is the file name for the smoothed mesh.

  The programs read and write meshes in ascii format.
*/

#include "../iss_io.h"

#include "stlib/geom/mesh/iss/laplacian.h"
#include "stlib/geom/mesh/iss/build.h"
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
         << "  [closestPoint] [-boundary=file] [-onlyBoundary=file]\n"
         << "  [-angle=maxDeviation] [-sweeps=n] in out\n"
         << "-closestPoint specifies that the closest point (instead of the\n"
         << " closest point in the normal direction) should be used.\n"
         << "-boundary specifies that the boundary should be smoothed as well.\n"
         << "-onlyBoundary specifies that only the boundary should be smoothed.\n"
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
   typedef geom::IndSimpSetIncAdj<Dimension, Dimension> ISS;
   typedef geom::IndSimpSetIncAdj < Dimension, Dimension - 1 > ISS_Boundary;
   typedef geom::IndSimpSet < Dimension, Dimension - 1 > Boundary;
   // The functor for computing signed distance.
   typedef geom::ISS_SignedDistance<Boundary> ISS_SD;

   ads::ParseOptionsArguments parser(argc, argv);

   programName = parser.getProgramName();

   //
   // Get the options.
   //

   // Use the closest point (instead of the closest point along the normal
   // direction) for the boundary condition.
   const bool areUsingClosestPoint = parser.getOption("closestPoint");

   // By default one sweep is performed.
   int numberOfSweeps = 1;
   parser.getOption("sweeps", &numberOfSweeps);

   std::string boundaryFileName;
   bool areSmoothingBoundary = false;
   bool areSmoothingInterior = true;
   if (parser.getOption("boundary", &boundaryFileName)) {
      areSmoothingBoundary = true;
   }
   else if (parser.getOption("onlyBoundary", &boundaryFileName)) {
      areSmoothingBoundary = true;
      areSmoothingInterior = false;
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
   std::cout << "Quality of the input mesh:\n";
   geom::printQualityStatistics(std::cout, mesh);

   ads::Timer timer;
   double elapsedTime;

   if (areSmoothingBoundary) {
      Boundary boundary;
      // Read the boundary file.
      readAscii(boundaryFileName.c_str(), &boundary);
      // The data structure and functor that computes the signed distance and
      // closest point.
      ISS_SD signedDistance(boundary);

      std::cout << "\nSmoothing the boundary vertices..." << std::flush;
      timer.tic();

      // Get the boundary of the mesh.
      ISS_Boundary meshBoundary;
      std::vector<std::size_t> indices;
      geom::buildBoundary(mesh, &meshBoundary, std::back_inserter(indices));

      if (areUsingClosestPoint) {
         // The functor that returns the closest point.
         geom::ISS_SD_ClosestPoint<Boundary> closestPoint(signedDistance);
         // Smooth the boundary.
         geom::applyLaplacian(&meshBoundary, closestPoint, maxAngleDeviation,
                              numberOfSweeps);
      }
      else {
         // The functor that returns the closest point.
         geom::ISS_SD_ClosestPointDirection<Boundary>
         closestPoint(signedDistance);
         // Smooth the boundary.
         geom::applyLaplacian(&meshBoundary, closestPoint, maxAngleDeviation,
                              numberOfSweeps);
      }

      // Transfer the new boundary vertex positions..
      for (std::size_t n = 0; n != indices.size(); ++n) {
         mesh.vertices[indices[n]] = meshBoundary.vertices[n];
      }

      elapsedTime = timer.toc();
      std::cout << "done.  Operatation took " << elapsedTime << " seconds.\n\n";
   }

   if (areSmoothingInterior) {
      std::cout << "\nSmoothing the interior vertices..." << std::flush;
      timer.tic();

      geom::applyLaplacian(&mesh, numberOfSweeps);

      elapsedTime = timer.toc();
      std::cout << "done.  Operatation took " << elapsedTime << " seconds.\n\n";
   }

   // Print quality measures for the output mesh.
   std::cout << "Quality of the smoothed mesh:\n";
   geom::printQualityStatistics(std::cout, mesh);

   // Write the output mesh.
   writeAscii(parser.getArgument().c_str(), mesh);

   // There should be no more arguments.
   assert(parser.areArgumentsEmpty());

   return 0;
}
