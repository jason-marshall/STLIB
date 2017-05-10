// -*- C++ -*-

/*!
  \file moveBoundary.h
  \brief Move the boundary vertices of an N-D mesh to lie on an (N-1)-D mesh.
*/

/*!
  \page examples_geom_mesh_moveBoundary Move the boundary vertices.


  This program moves the boundary vertices of a 2-D/3-D mesh to lie on
  a curve/surface.  It accepts a solid mesh and a boundary mesh as
  input.  The boundary mesh defines the curve/surface.  It moves the
  boundary vertices on the solid mesh to either their closest points on the
  boundary mesh or their closest points in the normal direction.
  It then outputs the resulting solid mesh.

  \verbatim
  moveBoundaryN.exe [-closestPoint] [-distance=max] boundary input output
  \endverbatim

  N may be either 2 or 3.
  - The closestPoint option specifies that the closest point (instead of the
    closest point in the normal direction) should be used.
  - distance specifies the maximum distance a point is allowed to move.
  - boundary is the boundary mesh.
  - input is the input solid mesh.
  - output is the transformed solid mesh.
*/

#include "../iss_io.h"

#include "stlib/geom/mesh/iss/ISS_SignedDistance.h"
#include "stlib/geom/mesh/iss/set.h"
#include "stlib/geom/mesh/iss/transform.h"
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
         << programName
         << " [-closestPoint] [-distance=max] boundary input output\n"
         << "- The closestPoint option specifies that the closest point (instead of the\n"
         << "  closest point in the normal direction) should be used.\n"
         << "- distance specifies the maximum distance a point is allowed to move.\n"
         << "- boundary is the boundary mesh.\n"
         << "- input is the input solid mesh.\n"
         << "- output is the transformed solid mesh.\n";
   exit(1);
}

}


//! The main loop.
int
main(int argc, char* argv[]) {
   static_assert(Dimension == 2 || Dimension == 3,
                 "The dimension must be 2 or 3.");

   typedef geom::IndSimpSetIncAdj<Dimension, Dimension> Mesh;
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
   bool areUsingClosestPoint = parser.getOption("closestPoint");

   // By default the maximum distance a point can move is infinity.
   double maxDistance = std::numeric_limits<double>::max();
   parser.getOption("distance", &maxDistance);
   if (maxDistance <= 0) {
      std::cerr << "Bad value for the maximim distance.\n";
      exitOnError();
   }

   // Check that we parsed all of the options.
   if (! parser.areOptionsEmpty()) {
      std::cerr << "Error.  Unmatched options:\n";
      parser.printOptions(std::cerr);
      exitOnError();
   }

   // If they did not specify the boundary, the input, and the output mesh.
   if (parser.getNumberOfArguments() != 3) {
      std::cerr << "Bad number of required arguments.\n"
                << "You gave the arguments:\n";
      parser.printArguments(std::cerr);
      exitOnError();
   }

   //
   // Read the input boundary mesh.
   //
   Boundary boundary;
   readAscii(parser.getArgument().c_str(), &boundary);

   std::cout << "The boundary mesh has " << boundary.vertices.size()
             << " vertices and " << boundary.indexedSimplices.size()
             << " simplices.\n";

   //
   // Read the input solid mesh.
   //
   Mesh mesh;
   readAscii(parser.getArgument().c_str(), &mesh);

   std::cout << "The input mesh has " << mesh.vertices.size()
             << " vertices and " << mesh.indexedSimplices.size()
             << " simplices.\n";


   std::cout << "Moving the boundary...\n" << std::flush;
   ads::Timer timer;
   timer.tic();

   // Get the boundary vertices.
   std::vector<std::size_t> bs;
   geom::determineBoundaryVertices(mesh, std::back_inserter(bs));
   // The data structure and functor that computes the signed distance.
   ISS_SD signedDistance(boundary);

   if (areUsingClosestPoint) {
      // The functor that returns the closest/closer point.
      geom::ISS_SD_CloserPoint<Boundary> cp(signedDistance, maxDistance);
      // Move the boundary vertices.
      geom::transform(&mesh, bs.begin(), bs.end(), cp);
   }
   else {
      // The functor that returns the closest/closer point along the
      // normal direction.
      geom::ISS_SD_CloserPointDirection<Boundary>
      cp(signedDistance, maxDistance);
      // Move the boundary vertices.
      geom::transform(&mesh, bs.begin(), bs.end(), cp);
   }

   double elapsedTime = timer.toc();
   std::cout << "done.\nTransform took " << elapsedTime
             << " seconds.\n";


   // Write the output mesh.
   writeAscii(parser.getArgument().c_str(), mesh);

   // There should be no more arguments.
   assert(parser.areArgumentsEmpty());

   return 0;
}
