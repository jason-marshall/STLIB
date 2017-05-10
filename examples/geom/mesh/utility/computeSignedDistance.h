// -*- C++ -*-

/*!
  \file computeSignedDistance.h
  \brief Computes cell attributes for simplicial meshes.
*/

/*!
  \page examples_geom_mesh_utility_computeSignedDistance Cell Attributes


  \section computeSignedDistanceIntroduction Introduction

  This program computes the signed distance to a simplicial mesh (and
  optionally the closest point on the mesh) for a set of points.

  \section computeSignedDistanceUsage Usage

  \verbatim
  computeSignedDistanceN.exe [-cp|closestPoint=file] inputMesh inputPoints outputDistance
  \endverbatim

  Here N is the space dimension.  The simplex dimension is N-1.

  The file format for the output distance follows.
  \verbatim
  numberOfPoints
  distance_0
  distance_1
  ...
  \endverbatim
*/

#include "../iss_io.h"

#include "stlib/geom/mesh/iss/ISS_SignedDistance.h"
#include "stlib/ads/timer/Timer.h"
#include "stlib/ads/utility/ParseOptionsArguments.h"

#include <cassert>

USING_STLIB_EXT_VECTOR_IO_OPERATORS;
namespace std {USING_STLIB_EXT_ARRAY_IO_OPERATORS;}
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
         << programName
         << " [-cp|closestPoint=file] inputMesh inputPoints outputDistance\n";
   exit(1);
}

}


//! The main loop.
int
main(int argc, char* argv[]) {
   // The mesh.
   typedef geom::IndSimpSet < Dimension, Dimension - 1 > ISS;
   // A Cartesian point.
   typedef ISS::Vertex Point;
   // The signed distance data structure.
   typedef geom::ISS_SignedDistance<ISS, Dimension> ISS_SD;

   ads::ParseOptionsArguments parser(argc, argv);

   programName = parser.getProgramName();

   // If they did not specify the input mesh, the input points and the
   // output array.
   if (parser.getNumberOfArguments() != 3) {
      std::cerr << "Bad number of required arguments.\n"
                << "You gave the arguments:\n";
      parser.printArguments(std::cerr);
      exitOnError();
   }

   // Read the input mesh.
   ISS mesh;
   readAscii(parser.getArgument().c_str(), &mesh);
   // Make the signed distance data structure.
   ISS_SD signedDistanceFunctor(mesh);

   // Read the input points.
   std::vector<Point> points;
   {
      std::ifstream in(parser.getArgument().c_str());
      if (! in) {
         std::cerr << "Bad input file for the points.  Exiting...\n";
         exitOnError();
      }
      in >> points;
   }
   std::cout << "There are " << points.size() << " input points.\n"
             << std::flush;

   // The array of signed distances.
   std::vector<double> signedDistance(points.size());

   // See if we should compute the closest points as well.
   bool areComputingClosestPoints = false;
   std::vector<Point> closestPoints;
   std::string closestPointsFileName;
   // If we should compute the closest points.
   if (parser.getOption("cp", &closestPointsFileName) ||
         parser.getOption("closestPoint", &closestPointsFileName)) {
      areComputingClosestPoints = true;
      closestPoints.resize(points.size());
   }

   // Check that we parsed all of the options.
   if (! parser.areOptionsEmpty()) {
      std::cerr << "Error.  Unmatched options:\n";
      parser.printOptions(std::cerr);
      exitOnError();
   }

   // Start the timer.
   std::cout << "Computing the signed distance.\n" << std::flush;
   ads::Timer timer;
   timer.tic();

   if (areComputingClosestPoints) {
      for (std::size_t n = 0; n != points.size(); ++n) {
         signedDistance[n] = signedDistanceFunctor(points[n], &closestPoints[n]);
      }
   }
   else {
      for (std::size_t n = 0; n != points.size(); ++n) {
         signedDistance[n] = signedDistanceFunctor(points[n]);
      }
   }

   // Record the elapsed time.
   const double elapsedTime = timer.toc();
   std::cout << "Elapsed time = " << elapsedTime << ".\n" << std::flush;

   // Write the signed distance.
   {
      // Open the signed distance output file.
      std::ofstream out(parser.getArgument().c_str());
      if (! out) {
         std::cerr << "Bad distance output file.  Exiting...\n";
         exitOnError();
      }

      // There should be no more arguments.
      assert(parser.areArgumentsEmpty());

      // Set the precision.
      out.precision(std::numeric_limits<double>::digits10);
      // Write the distance.
      out << signedDistance;
   }

   if (areComputingClosestPoints) {
      // Open the closest points output file.
      std::ofstream out(closestPointsFileName.c_str());
      if (! out) {
         std::cerr << "Bad closest points output file.  Exiting...\n";
         exitOnError();
      }

      // Set the precision.
      out.precision(std::numeric_limits<double>::digits10);
      // Write the closest points.
      out << closestPoints;
   }

   return 0;
}
