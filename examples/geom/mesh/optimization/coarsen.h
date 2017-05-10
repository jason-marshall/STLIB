// -*- C++ -*-

/*!
  \file coarsen.h
  \brief Coarsen the mesh by collapsing edges.
*/

/*!
  \page examples_geom_mesh_coarsen Coarsen the mesh by collapsing edges.






  \section examples_geom_mesh_coarsen_example2 2-D Example: A

  We start with a coarse mesh of the letter "A".  In the following figures,
  we show the modified condition number of the cells.

  \verbatim
  cp ../../../data/geom/mesh/22/a_23.txt mesh.txt
  \endverbatim

  \image html coarsen22AMesh.jpg "The initial mesh."
  \image latex coarsen22AMesh.pdf "The initial mesh."

  We refine the mesh so no edge is longer than 0.02.
  This mesh has 4096 triangles.

  \verbatim
  refine22.exe -length=0.02 mesh.txt refined.txt
  \endverbatim

  \image html coarsen22ARefined.jpg "The refined mesh."
  \image latex coarsen22ARefined.pdf "The refined mesh."

  Then we perform coarsening.  The algorith will try to collapse any edge with
  length less than 0.05.  For the boundary nodes, any exterior angle that
  deviates more than \f$\pi / 6\f$ from straight will be considered a
  corner feature.  For now, we do not place any restrictions on the quality
  of the elements resulting from collapsing the edges.

  \verbatim
  coarsen22.exe -length=0.05 -angle=0.52 refined.txt c1o0.txt
  \endverbatim

  \image html coarsen22AC1O0.jpg "The coarsened mesh."
  \image latex coarsen22AC1O0.pdf "The coarsened mesh."

  Next we apply topological and geometric optimization to the mesh.

  \verbatim
  flip22.exe -function=c c1o0.txt c1o1.txt
  geometricOptimize2.exe -function=c -angle=0.52 c1o1.txt c1o1.txt
  \endverbatim

  \image html coarsen22AC1O1.jpg "One cycle of coarsening and optimization."
  \image latex coarsen22AC1O1.pdf "One cycle of coarsening and optimization."

  We apply one more cycle of coarsening and optimization to obtain a high
  quality mesh with the desired edge lengths.

  \image html coarsen22AC2O2.jpg "Two cycles of coarsening and optimization."
  \image latex coarsen22AC2O2.pdf "Two cycles of coarsening and optimization."

  We can also coarsen the mesh using a quality restriction on the edge
  collapses.  We specify that elements resulting from a collapse may not have
  a modified condition number less than 0.4.  In the figure below we see that
  the coarsening does not make as much progress as before.

  \verbatim
  coarsen22.exe -length=0.05 -angle=0.52 -function=c -minimumQuality=0.4 refined.txt c1o0q.txt
  \endverbatim

  \image html coarsen22AC1O0Q.jpg "The coarsened mesh using a quality restriction."
  \image latex coarsen22AC1O0Q.pdf "The coarsened mesh using a quality restriction."

  As before, we apply topological and geometric optimization.

  \image html coarsen22AC1O1Q.jpg "One cycle of coarsening and optimization using a quality restriction."
  \image latex coarsen22AC1O1Q.pdf "One cycle of coarsening and optimization using a quality restriction."

  One more cycle of coarsening and optimization yields the desired result.
  Because this is such an easy problem, using a quality restriction on the
  coarsen did not provide any benefit.

  \image html coarsen22AC2O2Q.jpg "Two cycles of coarsening and optimization using a quality restriction."
  \image latex coarsen22AC2O2Q.pdf "Two cycles of coarsening and optimization using a quality restriction."
*/

#include "../smr_io.h"
#include "../iss_io.h"

#include "stlib/geom/mesh/simplicial/coarsen.h"
#include "stlib/geom/mesh/simplicial/quality.h"
#include "stlib/ads/timer/Timer.h"
#include "stlib/ads/utility/ParseOptionsArguments.h"
#include "stlib/ads/functor/constant.h"

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
         << "  [-noBoundary] [-length=length] [-sweeps=max]\n"
         << "  [-manifold=manifold] [-featureDistance=d]\n"
         << "  [-function=f] [-minimumQuality=m] [-qualityFactor=q]\n"
#if SPACE_DIMENSION == 2 && SIMPLEX_DIMENSION == 2
         << "  [-angle=maxAngleDeviation] \n"
#elif SPACE_DIMENSION == 3 && SIMPLEX_DIMENSION == 3
         << "  [-dihedralAngle=maxDihedralAngleDeviation] \n"
         << "  [-solidAngle=maxSolidAngleDeviation] \n"
#else
         << "  [-dihedralAngle=maxDihedralAngleDeviation] \n"
         << "  [-solidAngle=maxSolidAngleDeviation] \n"
         << "  [-boundaryAngle=maxBoundaryAngleDeviation] \n"
#endif
         << "  fine coarse\n"
         << "-noBoundary specifies that the boundary should not be modified.\n"
         << "-length is used to specify the minimum edge length.\n"
         << "   If no length is specified, the minimum edge length will be set to\n"
         << "   1.01 times the average input edge length.\n"
         << "-sweeps specifies the maximum number of sweeps over the cells.\n"
         << "   By default the number of sweeps is not limited.\n"
         << "- The function specifies the quality function to use in determining\n"
         << "    if an edge collapse is allowed.\n"
         << "- An edge collapse is not allowed if it results in incident cells\n"
         << "    that have quality lower that the specified minimum.  By default,\n"
         << "    this minimum is zero.\n"
         << "- An edge collapse is allowed only if the quality of the incident\n"
         << "-   cells satisfy:\n"
         << "-   newQuality >= qualityFactor * oldQuality\n"
#if SPACE_DIMENSION == 2 && SIMPLEX_DIMENSION == 2
         << "- The angle is used to determine corner features.\n"
#elif SPACE_DIMENSION == 3 && SIMPLEX_DIMENSION == 3
         << "- The dihedral angle is used to determine edge features.\n"
         << "- The solid angle is used to determine corner features.\n"
#else
         << "- The dihedral angle is used to determine edge features.\n"
         << "- The solid angle is used to determine corner features.\n"
         << "- The boundary angle is used to determine boundary corner features.\n"
#endif
         << "fine is the input mesh.\n"
         << "coarse is the output mesh.\n";
   exit(1);
}

}


//! The main loop.
int
main(int argc, char* argv[]) {
   static_assert((SPACE_DIMENSION == 2 && SIMPLEX_DIMENSION == 2) ||
                 (SPACE_DIMENSION == 3 && SIMPLEX_DIMENSION == 2) ||
                 (SPACE_DIMENSION == 3 && SIMPLEX_DIMENSION == 3),
                 "Those dimensions are not supported.");

#if SPACE_DIMENSION == SIMPLEX_DIMENSION
   typedef geom::IndSimpSet<SPACE_DIMENSION, SIMPLEX_DIMENSION - 1> BoundaryMesh;
   typedef geom::PointsOnManifold<SPACE_DIMENSION, SIMPLEX_DIMENSION - 1, 1>
      BoundaryManifold;
#else
   typedef geom::IndSimpSet<SPACE_DIMENSION, SIMPLEX_DIMENSION> BoundaryMesh;
   typedef geom::PointsOnManifold<SPACE_DIMENSION, SIMPLEX_DIMENSION, 1>
      BoundaryManifold;
#endif
   typedef geom::SimpMeshRed<SPACE_DIMENSION, SIMPLEX_DIMENSION> Mesh;
   typedef Mesh::Vertex Vertex;

   // The possible quality functions.
   enum QualityFunction {ConditionNumber, MeanRatio};

   ads::ParseOptionsArguments parser(argc, argv);

   programName = parser.getProgramName();

   //
   // Get the options.
   //

   // Preserve the boundary.
   const bool arePreservingBoundary = parser.getOption("noBoundary");

   // The minimum edge length.
   double length = 0;
   parser.getOption("length", &length);

   // The maximum number of sweeps.
   int maximumSweeps = 0;
   parser.getOption("sweeps", &maximumSweeps);

   double featureDistance = 0;
   if (parser.getOption("featureDistance", &featureDistance)) {
      if (featureDistance <= 0) {
         std::cerr << "Bad feature distance.\n";
         exitOnError();
      }
   }

   // The minimum allowed quality of an incident cell for an edge collapse to
   // be allowed.
   double minimumQuality = 0;
   parser.getOption("minimumQuality", &minimumQuality);

   // An edge collapse will be allowed only if
   // newQuality >= qualityFactor * oldQuality.
   double qualityFactor = 0;
   parser.getOption("qualityFactor", &qualityFactor);

   // The default function is condition number.
   QualityFunction function = ConditionNumber;
   char functionCharacter;
   if (parser.getOption("function", &functionCharacter)) {
      if (functionCharacter == 'c') {
         function = ConditionNumber;
      }
      else if (functionCharacter == 'm') {
         function = MeanRatio;
      }
      else {
         std::cerr << "Unrecognized function.\n";
         exitOnError();
      }
   }

   // If they did not specify the input and output mesh.
   if (parser.getNumberOfArguments() != 2) {
      std::cerr << "Bad number of required arguments.\n"
                << "You gave the arguments:\n";
      parser.printArguments(std::cerr);
      exitOnError();
   }

   // Read the input mesh.
   Mesh mesh;
   readAscii(parser.getArgument().c_str(), &mesh);

   // The boundary mesh.
   std::string boundaryFile;
   BoundaryManifold* boundaryManifold = 0;

   //
   // If we are not preserving the boundary, make the boundary manifold.
   //
   if (! arePreservingBoundary) {
      //
      // If a boundary manifold has been specified, read it.  Otherwise, build
      // it.
      //
      BoundaryMesh boundaryMesh;
      if (parser.getOption("manifold", &boundaryFile)) {
         // Read the mesh describing the boundary.
         readAscii(boundaryFile.c_str(), &boundaryMesh);
      }
      else {
#if SPACE_DIMENSION == SIMPLEX_DIMENSION
         // Extract the boundary of the mesh.
         geom::IndSimpSetIncAdj<SPACE_DIMENSION, SIMPLEX_DIMENSION> iss;
         buildIndSimpSetFromSimpMeshRed(mesh, &iss);
         buildBoundary(iss, &boundaryMesh);
#else
         // Use the surface mesh as the boundary mesh.
         buildIndSimpSetFromSimpMeshRed(mesh, &boundaryMesh);
#endif
      }

      //
      // Build the boundary manifold data structure.
      //
#if SPACE_DIMENSION == 2 && SIMPLEX_DIMENSION == 2
      // Get the angle deviation for corner features.
      double maxAngleDeviation = -1;
      parser.getOption("angle", &maxAngleDeviation);
      // If they specify an angle for determining corner features.
      if (maxAngleDeviation >= 0) {
         // Build the boundary manifold.
         boundaryManifold = new BoundaryManifold(boundaryMesh, maxAngleDeviation);
         // If the feature distance was specified.
         if (featureDistance != 0) {
            // Set the maximum corner distance.
            boundaryManifold->setMaxCornerDistance(featureDistance);
         }
         // Register the boundary vertices of the mesh.
         boundaryManifold->insertBoundaryVertices(&mesh);
      }
#elif SPACE_DIMENSION == 3 && SIMPLEX_DIMENSION == 3
      double maxDihedralAngleDeviation = -1;
      parser.getOption("dihedralAngle", &maxDihedralAngleDeviation);
      double maxSolidAngleDeviation = -1;
      parser.getOption("solidAngle", &maxSolidAngleDeviation);
      // If they specified angles for determining features.
      if (maxDihedralAngleDeviation >= 0 || maxSolidAngleDeviation >= 0) {
         // Build the boundary manifold.
         boundaryManifold = new BoundaryManifold(boundaryMesh,
                                                 maxDihedralAngleDeviation,
                                                 maxSolidAngleDeviation);
         // If the feature distance was specified.
         if (featureDistance != 0) {
            // Set the maximum corner distance.
            boundaryManifold->setMaxCornerDistance(featureDistance);
            // Set the maximum edge distance.
            boundaryManifold->setMaxEdgeDistance(featureDistance);
         }
         // Register the boundary vertices and edges of the mesh.
         boundaryManifold->insertBoundaryVerticesAndEdges
         (&mesh, maxDihedralAngleDeviation);
      }
#else
      double maxDihedralAngleDeviation = -1;
      parser.getOption("dihedralAngle", &maxDihedralAngleDeviation);
      double maxSolidAngleDeviation = -1;
      parser.getOption("solidAngle", &maxSolidAngleDeviation);
      double maxBoundaryAngleDeviation = -1;
      parser.getOption("boundaryAngle", &maxBoundaryAngleDeviation);
      // If they specified angles for determining features.
      if (maxDihedralAngleDeviation >= 0 || maxSolidAngleDeviation >= 0 ||
            maxBoundaryAngleDeviation >= 0) {
         // Build the boundary manifold.
         boundaryManifold = new BoundaryManifold(boundaryMesh,
                                                 maxDihedralAngleDeviation,
                                                 maxSolidAngleDeviation,
                                                 maxBoundaryAngleDeviation);
         // If the feature distance was specified.
         if (featureDistance != 0) {
            // Set the maximum corner distance.
            boundaryManifold->setMaxCornerDistance(featureDistance);
            // Set the maximum edge distance.
            boundaryManifold->setMaxEdgeDistance(featureDistance);
         }
         // Register the vertices and edges of the surface mesh.
         boundaryManifold->insertVerticesAndEdges(&mesh,
               maxDihedralAngleDeviation);
      }
#endif
   }

   // Check that we parsed all of the options.
   if (! parser.areOptionsEmpty()) {
      std::cerr << "Error.  Unmatched options:\n";
      parser.printOptions(std::cerr);
      exitOnError();
   }

   // Print quality measures for the fine mesh.
   std::cout << "Quality of the input mesh:\n";
   printQualityStatistics(std::cout, mesh);
   std::cout << "Number of edges = " << mesh.computeEdgesSize() << "\n";
   geom::printEdgeLengthStatistics(std::cout, mesh);

   if (length == 0) {
      double minLength, maxLength, meanLength;
      geom::computeEdgeLengthStatistics(mesh, &minLength, &maxLength,
                                        &meanLength);
      length = 1.01 * meanLength;
      std::cout << "\nWill try to collapse edges shorter than " << length
                << ".\n";
   }

   // Print information about the manifold data structure.
   if (boundaryManifold != 0) {
      std::cout << "\n";
      boundaryManifold->printInformation(std::cout);
      std::cout << "\n";
   }

   std::cout << "Coarsening the mesh...\n" << std::flush;
   ads::Timer timer;
   timer.tic();

   // Coarsen the mesh.
   int count = 0;
   if (arePreservingBoundary) {
#if SPACE_DIMENSION == SIMPLEX_DIMENSION
      if (function == ConditionNumber) {
         count = geom::coarsenInterior
                 <geom::SimplexModCondNum<SIMPLEX_DIMENSION> >
                 (&mesh,
                  ads::constructUnaryConstant<Vertex, double>(length),
                  minimumQuality, qualityFactor,
                  boundaryManifold, maximumSweeps);
      }
      else if (function == MeanRatio) {
         count = geom::coarsenInterior
                 <geom::SimplexModMeanRatio<SIMPLEX_DIMENSION> >
                 (&mesh,
                  ads::constructUnaryConstant<Vertex, double>(length),
                  minimumQuality, qualityFactor,
                  boundaryManifold, maximumSweeps);
      }
      else {
         assert(false);
      }
#else
      std::cerr << "Sorry.  This option is not supported.\n";
#endif
   }
   else {
      if (function == ConditionNumber) {
         count = geom::coarsen<geom::SimplexModCondNum<SIMPLEX_DIMENSION> >
                 (&mesh,
                  ads::constructUnaryConstant<Vertex, double>(length),
                  minimumQuality, qualityFactor,
                  boundaryManifold, maximumSweeps);
      }
      else if (function == MeanRatio) {
         count = geom::coarsen<geom::SimplexModMeanRatio<SIMPLEX_DIMENSION> >
                 (&mesh,
                  ads::constructUnaryConstant<Vertex, double>(length),
                  minimumQuality, qualityFactor,
                  boundaryManifold, maximumSweeps);
      }
      else {
         assert(false);
      }
   }

   double elapsedTime = timer.toc();
   std::cout << "done.\n"
             << "Number of edges collapsed = " << count << "\n"
             << "Coarsening took " << elapsedTime << " seconds.\n";

   // Print quality measures for the coarsened mesh.
   std::cout << "\nQuality of the coarsened mesh:\n";
   geom::printQualityStatistics(std::cout, mesh);
   std::cout << "Number of edges = " << mesh.computeEdgesSize() << "\n";
   geom::printEdgeLengthStatistics(std::cout, mesh);

   // Write the output mesh.
   writeAscii(parser.getArgument().c_str(), mesh);

   // There should be no more arguments.
   assert(parser.areArgumentsEmpty());

   // Delete the boundary manifold if it was allocated.
   if (boundaryManifold != 0) {
      delete boundaryManifold;
      boundaryManifold = 0;
   }

   return 0;
}
