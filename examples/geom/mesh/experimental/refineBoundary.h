// -*- C++ -*-

/*!
  \file refineBoundary.h
  \brief Refine the boundary cells to more closely match a curve/surface.
*/

/*!
  \page examples_geom_mesh_refineBoundary Refine the boundary cells to more closely match a curve/surface.

  This program accepts a 2-D/3-D solid mesh and a curve/surface as input.
  It refines boundary cells where they do not match the curve/surface well.
  This produces new boundary vertices that are moved to lie on the
  curve/surface.

  Currently refining is only implemented for 2-D meshes.  The algorithm
  for 3-D refining is essentially the same and will be implemented shortly.



  \section examples_geom_mesh_refineBoundary_example2 2-D Example

  We start with a coarse mesh that approximately fills a specified boundary.
  The edge lengths are each about 0.04.
  This mesh has 190 vertices and 263 triangles.  The minimum condition
  number is 0.46; the mean is 0.95.

  \verbatim
  cp ../../../data/geom/mesh/21/a.txt boundary.txt
  tile2.exe -smooth -move -length=0.04 boundary.txt mesh.txt
  \endverbatim

  \image html refine_boundary_2_boundary.jpg "The boundary."
  \image latex refine_boundary_2_boundary.pdf "The boundary."

  \image html refine_boundary_2_mesh.jpg "The coarse mesh."
  \image latex refine_boundary_2_mesh.pdf "The coarse mesh."

  We apply refinement at the boundary.  We refine any cell with an
  edge on the boundary longer than 0.01 and that deviates from the specified
  boundary by an angle of more that 0.1 radians.  The refinement stays local;
  much of the mesh is unchanged.  Unfortunately, the resulting mesh has
  poor quality.
  It has 315 vertices and 478 triangles.  The minimum condition
  number is 3.7e-10; the mean is 0.85.  There are four inverted triangles.

  \verbatim
  refineBoundary2.exe -angle=0.1 -length=0.01 mesh.txt boundary.txt mesh_r.txt
  \endverbatim

  \image html refine_boundary_2_mesh_r.jpg "The mesh after refining at the boundary."
  \image latex refine_boundary_2_mesh_r.pdf "The mesh after refining at the boundary."

  We obtain better results if we perform edge flipping and geometric
  optimization after each sweep of edge refinement.  The operations keep
  the boundary condition from distorting the triangles during repeated
  refinement.  The geometric optimization causes a "diffusion" of the
  refinement, but it still stays local.
  Now the mesh has 347 vertices and 532 triangles.  The minimum condition
  number is 0.48; the mean is 0.93.

  \verbatim
  refineBoundary2.exe -flip -smooth -angle 0.1 -length=0.01 mesh.txt boundary.txt mesh_rfs.txt
  \endverbatim

  \image html refine_boundary_2_mesh_rfs.jpg "Boundary refinement with edge flipping and geometric optimization."
  \image latex refine_boundary_2_mesh_rfs.pdf "Boundary refinement with edge flipping and geometric optimization."
*/

#ifndef DIMENSION
#error DIMENSION must be defined to compile this program.
#endif

#include "../smr_io.h"
#include "../iss_io.h"

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
         << " [-flip] [-smooth] [-angle=max] [-length=min] [-sweeps=f] [-function=f]"
         << " boundary input output\n"
         << "- angle specifies the maximum allowed angle between an edge and the boundary.\n"
         << "- length specifies the minimum allowed edge length.\n"
         << "- sweeps specifies the maximum number of sweeps.\n"
         << "- -flif indicates that flipping will be done following each sweep.\n"
         << "- -smooth indicates that smoothing will be done following each sweep.\n"
         << "- -function lets you choose the quality function.\n"
         << "  The method should be either 'm' for mean ratio or\n"
         << "  'c' for condition number.  By default it is mean ratio.\n"
         << "- boundary is the boundary mesh.\n"
         << "- input is the input solid mesh.\n"
         << "- output is the transformed solid mesh.\n";
   exit(1);
}

}


//! The main loop.
int
main(int argc, char* argv[]) {
   // CONTINUE
   static_assert(DIMENSION == 2, "The dimension must be 2.");

   typedef geom::SimpMeshRed<DIMENSION, DIMENSION> Mesh;
   typedef geom::IndSimpSetIncAdj<DIMENSION, DIMENSION> ISS;
   typedef geom::IndSimpSet < DIMENSION, DIMENSION - 1 > Boundary;
   // The functor for computing signed distance and closest point.
   typedef geom::ISS_SignedDistance<Boundary> ISS_SD;
   // The functor for computing the closest point.
   typedef geom::ISS_SD_ClosestPointDirection<Boundary> ISS_SD_CP;

   enum Method {MeanRatio, ConditionNumber};

   ads::ParseOptionsArguments parser(argc, argv);

   programName = parser.getProgramName();

   //
   // Get the options.
   //

   double maximumAngle = 10.0 * numerical::Constants<double>::Degree();
   parser.getOption("angle", &maximumAngle);
   if (maximumAngle <= 0) {
      std::cerr << "Bad value for maximum angle.\n";
      exitOnError();
   }

   double minimumLength = 0.0;
   parser.getOption("length", &minimumLength);
   if (minimumLength < 0) {
      std::cerr << "Bad value for minimum length.\n";
      exitOnError();
   }

   int maximumSweeps = 10;
   parser.getOption("sweeps", &maximumSweeps);
   if (maximumSweeps <= 0) {
      std::cerr << "Bad value for maximum number of sweeps.\n";
      exitOnError();
   }

   bool flip = parser.getOption("flip");

   bool smooth = parser.getOption("smooth");

   // The default method is mean ratio.
   Method method = MeanRatio;
   std::string functionName;
   if (parser.getOption("function", &functionName)) {
      if (functionName == "m") {
         method = MeanRatio;
      }
      else if (functionName == "c") {
         method = ConditionNumber;
      }
      else {
         std::cerr << "Bad function.\n";
         exitOnError();
      }
   }

   // Check that we parsed all of the options.
   if (! parser.areOptionsEmpty()) {
      std::cerr << "Error.  Unmatched options:\n";
      parser.printOptions(std::cerr);
      exitOnError();
   }

   // If they did not specify the boundary, the input and the output mesh.
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
   std::cout << "The boundary mesh has " << boundary.getVerticesSize()
             << " vertices and " << boundary.getSimplicesSize()
             << " simplices.\n";

   //
   // Read the input solid mesh.
   //
   Mesh mesh;
   readAscii(parser.getArgument().c_str(), &mesh);
   std::cout << "The input mesh has " << mesh.computeNodesSize()
             << " vertices and " << mesh.computeCellsSize()
             << " simplices.\n";

   std::cout << "Refining the boundary...\n" << std::flush;
   ads::Timer timer;
   timer.tic();

   int numberOfEdgesSplit = 0;
   if (flip || smooth) {
      // The data structure and functor that computes the signed distance.
      ISS_SD signedDistance(boundary);
      // The functor that returns the closest point.
      ISS_SD_CP closestPoint(signedDistance);

      int sweep = 0;
      int n;
      do {
         n = geom::refineBoundary(&mesh, boundary, maximumAngle, minimumLength,
                                  1);
         numberOfEdgesSplit += n;

         if (flip) {
            if (method == MeanRatio) {
               flipUsingModifiedMeanRatio(&mesh);
            }
            else if (method == ConditionNumber) {
               flipUsingModifiedConditionNumber(&mesh);
            }
            else {
               assert(false);
            }
         }

         if (smooth) {
            // Make an indexed simplex set.
            ISS iss;
            geom::buildIndSimpSetFromSimpMeshRed(mesh, &iss);

            // The set of interior vertices.
            std::vector<std::size_t> iv;
            geom::determineInteriorVertices(iss, std::back_inserter(iv));
            // The set of boundary vertices.
            std::vector<std::size_t> bv;
            geom::determineComplementSetOfIndices(iss.getVerticesSize(),
                                                  iv.begin(), iv.end(),
                                                  std::back_inserter(bv));

            if (method == MeanRatio) {
               // Optimize the interior vertices.
               geom::geometricOptimizeUsingMeanRatio(&iss, iv.begin(), iv.end());
               // Optimize the boundary vertices.
               geom::geometricOptimizeWithConditionUsingMeanRatio
               (&iss, bv.begin(), bv.end(), closestPoint);
            }
            else if (method == ConditionNumber) {
               // Optimize the interior vertices.
               geom::geometricOptimizeUsingConditionNumber(&iss,
                     iv.begin(), iv.end());
               // Optimize the boundary vertices.
               geom::geometricOptimizeWithConditionUsingConditionNumber
               (&iss, bv.begin(), bv.end(), closestPoint);
            }
            else {
               assert(false);
            }

            // Transfer the new positions to the mesh.
            mesh.setVertices(iss.getVerticesBeginning(), iss.getVerticesEnd());
         }

         ++sweep;
      }
      while (sweep != maximumSweeps && n != 0);
   }
   else { // Neither flipping nor smoothing.
      numberOfEdgesSplit = geom::refineBoundary(&mesh, boundary, maximumAngle,
                           minimumLength, maximumSweeps);
   }

   double elapsedTime = timer.toc();
   std::cout << "done.\nRefinement took " << elapsedTime << " seconds.\n"
             << numberOfEdgesSplit << " edges were split.\n";


   // Write the output mesh.
   writeAscii(parser.getArgument().c_str(), mesh);

   // There should be no more arguments.
   assert(parser.areArgumentsEmpty());

   return 0;
}
