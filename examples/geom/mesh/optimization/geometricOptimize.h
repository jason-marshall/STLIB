// -*- C++ -*-

/*!
  \file geometricOptimize.h
  \brief Optimization of vertex locations with a boundary manifold constraint.
*/

/*!
  \page examples_geom_mesh_geometricOptimize Geometric Optimization of Simplicial Meshes

  \section examples_geom_mesh_geometricOptimize_introduction Introduction

  This program uses geometric optimization in conjuction with a boundary
  condition.  For interior vertices, the vertex is move to optimize the
  the 2-norm of the quality of the incident simplices.  For boundary
  vertices, the position is optimized on the boundary manifold.  Vertices
  which are on a corner feature may not be moved.  Vertices on edge features
  are moved along the edge feature.  Finally, vertices on surface features
  move on the surface, but may not cross edges.

  \section examples_geom_mesh_geometricOptimize_example3 Example in 3-D.

  We start with a mesh of the unit cube.
  The edges of the tetrahedra have lengths close to 0.1.  The cube is
  initially centered at the origin.  We move it into the first octant.
  In the figures below, we show the modified condition number of the elements.

  \verbatim
  cp ../../../data/geom/mesh/33/cube_1_1_1_0.1.txt mesh.txt
  python.exe ../../../data/geom/mesh/utilities/vertices_map.py translate mesh.txt mesh.txt
  \endverbatim

  \image html cubeSineMesh.jpg "The initial mesh of the cube.  The modified minimum condition number is 0.58; the mean is 0.86."
  \image latex cubeSineMesh.pdf "The initial mesh of the cube.  The modified minimum condition number is 0.58; the mean is 0.86."

  We distort the mesh by moving the vertices according to the following
  function:

  \verbatim
  x = x + sin(13 pi (x + 0.5 sin(2 pi x) sin(2 pi (y + z)))) / 30
  y = y + sin(17 pi (y + 0.5 sin(2 pi y) sin(2 pi (z + x)))) / 30
  z = z + sin(19 pi (z + 0.5 sin(2 pi z) sin(2 pi (x + y)))) / 30
  \endverbatim

  For boundary vertices, this moves along the boundary, but not off the
  boundary.  The distortion yields a poor quality mesh.  162 tetrahedra
  have negative volume.

  \verbatim
  python.exe ../../../data/geom/mesh/utilities/vertices_map.py distort mesh.txt distorted.txt
  \endverbatim

  \image html cubeSineDistorted.jpg "The distorted mesh.  The modified minimum condition number is 7.2e-12; the mean is 0.62."
  \image latex cubeSineDistorted.pdf "The distorted mesh.  The modified minimum condition number is 7.2e-12; the mean is 0.62."

  We apply sweeps of geometric optimization to improve the quality of the
  distorted mesh.  We use a dihedral angle deviation of 0.5 to define edge
  features on the boundary.  Intersecting edge features form corner features.

  \verbatim
  boundary33.exe mesh.txt boundary.txt
  geometricOptimize3.exe -boundary=boundary.txt -dihedralAngle=0.5 distorted.txt optimized1.txt
  geometricOptimize3.exe -boundary=boundary.txt -dihedralAngle=0.5 optimized1.txt optimized2.txt
  geometricOptimize3.exe -boundary=boundary.txt -dihedralAngle=0.5 optimized2.txt optimized3.txt
  \endverbatim

  \image html cubeSineOptimized1.jpg "1 optimization sweep.  The minimum modified condition number is 3.0e-10; the mean is 0.84."
  \image latex cubeSineOptimized1.pdf "1 optimization sweep.  The minimum modified condition number is 3.0e-10; the mean is 0.84."

  \image html cubeSineOptimized2.jpg "2 optimization sweeps.  The minimum modified condition number is 0.44; the mean is 0.88."
  \image latex cubeSineOptimized2.pdf "2 optimization sweeps.  The minimum modified condition number is 0.44; the mean is 0.88."

  \image html cubeSineOptimized3.jpg "3 optimization sweeps.  The minimum modified condition number is 0.55; the mean is 0.89."
  \image latex cubeSineOptimized3.pdf "3 optimization sweeps.  The minimum modified condition number is 0.55; the mean is 0.89."

  Three sweeps of geometric optimization is sufficient to regain a quality
  mesh.  Since vertices along the edges of the cube are moved along the edges
  and vertices at corner features cannot be moved, the boundary of the
  mesh retains the same shape.
*/

#include "../iss_io.h"

#include "stlib/geom/mesh/iss/build.h"
#include "stlib/geom/mesh/iss/optimize.h"
#include "stlib/ads/timer/Timer.h"
#include "stlib/ads/utility/ParseOptionsArguments.h"

#include <string>
#include <sstream>

#include <cassert>

using namespace stlib;

namespace {

// Types.

//! The simplicial mesh.
typedef geom::IndSimpSetIncAdj<DIMENSION, DIMENSION> Mesh;
//! The boundary of the simplicial mesh.
typedef geom::IndSimpSet < DIMENSION, DIMENSION - 1 > Boundary;
//! The boundary manifold data structure.
typedef geom::PointsOnManifold < DIMENSION, DIMENSION - 1, 1 > BoundaryManifold;

// Enumerations.

// CONTINUE: Doxygen has a problem with enumerations. Documenting this
// will conflict with other enumerations.
// Quality functions.
enum Function {MeanRatio, ConditionNumber};

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
         << " [-noBoundary] [-function=f] [-sweeps=n] [-boundary=b] [-featureDistance=d]\n"
#if DIMENSION == 2
         << " [-angle=a]\n"
#elif DIMENSION == 3
         << " [-dihedralAngle=d] [-solidAngle=s]\n"
#endif
         << " input output\n\n"
         << "- noBoundary indicates that the boundary vertices should not be moved.\n"
         << "- boundary is used to specify a file name for the boundary mesh.\n"
         << "  If not specified, the boundary of the input mesh will be used.\n"
         << "- The function should be either 'm' for mean ratio or\n"
         << "  'c' for condition number.  By default it is mean ratio.\n"
         << "- The -sweeps option lets you specify the number of sweeps.\n"
#if DIMENSION == 2
         << "- angle is the maximum angle deviation (from pi)\n"
#elif DIMENSION == 3
         << "- dihedralAngle is the maximum dihedral angle deviation\n"
         << "  (from straight) for a surface feature.  The rest are edge "
         << "features.\n"
         << "- Solid angles that deviate more than maxSolidAngleDeviation\n"
         << "  (from 2*pi) are corner features.\n"
         << "  between two boundary edges for a corner feature on the boundary.\n"
#endif
         << "- The feature distance determines how close a vertex must be to a \n"
         << "  corner or edge to be placed on that feature.\n"
         << "- input is the file name for the input mesh.\n"
         << "- output is the file name for the optimized mesh.\n";
   exit(1);
}

}


//! The main loop.
int
main(int argc, char* argv[]) {
   static_assert(DIMENSION == 2 || DIMENSION == 3,
                 "The dimension must be 2 or 3.");

   ads::ParseOptionsArguments parser(argc, argv);

   programName = parser.getProgramName();

   //
   // Get the options.
   //

   const bool areOptimizingBoundary = ! parser.getOption("noBoundary");

   // The default function is mean ratio.
   Function function = MeanRatio;
   char functionCharacter;
   if (parser.getOption("function", &functionCharacter)) {
      if (functionCharacter == 'm') {
         function = MeanRatio;
      }
      else if (functionCharacter == 'c') {
         function = ConditionNumber;
      }
      else {
         std::cerr << "Unrecognized function.\n";
         exitOnError();
      }
   }

   // By default one sweep is performed.
   int numSweeps = 1;
   parser.getOption("sweeps", &numSweeps);
   if (numSweeps < 1) {
      std::cerr << "Bad number of sweeps.  You entered " << numSweeps << "\n";
      exitOnError();
   }

#if DIMENSION == 2
   double maxAngleDeviation = -1;
   parser.getOption("angle", &maxAngleDeviation);
#elif DIMENSION == 3
   double maxDihedralAngleDeviation = -1;
   parser.getOption("dihedralAngle", &maxDihedralAngleDeviation);
   double maxSolidAngleDeviation = -1;
   parser.getOption("solidAngle", &maxSolidAngleDeviation);
#endif

   double featureDistance = 0;
   if (parser.getOption("featureDistance", &featureDistance)) {
      if (featureDistance <= 0) {
         std::cerr << "Bad feature distance.\n";
         exitOnError();
      }
   }

   // If they did not specify the input mesh and output mesh.
   if (parser.getNumberOfArguments() != 2) {
      std::cerr << "Bad number of required arguments.\n"
                << "You gave the arguments:\n";
      parser.printArguments(std::cerr);
      exitOnError();
   }

   // Read the input solid mesh.
   Mesh mesh;
   readAscii(parser.getArgument().c_str(), &mesh);
   std::cout << "The solid mesh has " << mesh.vertices.size()
             << " vertices and " << mesh.indexedSimplices.size()
             << " simplices.\n";

   // The boundary manifold (initialized to null).
   BoundaryManifold* boundaryManifold = 0;
   if (areOptimizingBoundary) {
      //
      // Read or make the boundary.
      //
      std::string boundaryFileName;
      Boundary boundary;
      if (parser.getOption("boundary", &boundaryFileName)) {
         // Read the input boundary mesh.
         readAscii(boundaryFileName.c_str(), &boundary);
      }
      else {
         // Extract the boundary of the mesh.
         buildBoundary(mesh, &boundary);
      }
      std::cout << "The boundary mesh has " << boundary.vertices.size()
                << " vertices and " << boundary.indexedSimplices.size()
                << " simplices.\n";
      //
      // Build the boundary manifold.
      //
#if DIMENSION == 2
      boundaryManifold = new BoundaryManifold(boundary, maxAngleDeviation);
      // If the feature distance was specified.
      if (featureDistance != 0) {
         // Set the maximum corner distance.
         boundaryManifold->setMaxCornerDistance(featureDistance);
      }
      // Register points at each of the boundary vertices.
      boundaryManifold->insertBoundaryVertices(&mesh);
#elif DIMENSION == 3
      boundaryManifold = new BoundaryManifold(boundary,
                                              maxDihedralAngleDeviation,
                                              maxSolidAngleDeviation);
      // If the feature distance was specified.
      if (featureDistance != 0) {
         // Set the maximum corner distance.
         boundaryManifold->setMaxCornerDistance(featureDistance);
         // Set the maximum edge distance.
         boundaryManifold->setMaxEdgeDistance(featureDistance);
      }
      // Register points and edges of the boundary.
      boundaryManifold->
      insertBoundaryVerticesAndEdges(&mesh, maxDihedralAngleDeviation);
#endif
   }

   // Check that we parsed all of the options.
   if (! parser.areOptionsEmpty()) {
      std::cerr << "Error.  Unmatched options:\n";
      parser.printOptions(std::cerr);
      exitOnError();
   }

   // Print quality measures for the input mesh.
   geom::printQualityStatistics(std::cout, mesh);
   std::cout << "\n";
   // Print information about the manifold data structure.
   if (boundaryManifold != 0) {
      boundaryManifold->printInformation(std::cout);
      std::cout << "\n";
   }

   std::cout << "Optimizing the mesh...\n" << std::flush;
   ads::Timer timer;
   timer.tic();

   // Geometric optimization.
   if (function == MeanRatio) {
      geom::geometricOptimizeUsingMeanRatio(&mesh, boundaryManifold, numSweeps);
   }
   else if (function == ConditionNumber) {
      geom::geometricOptimizeUsingConditionNumber(&mesh, boundaryManifold,
            numSweeps);
   }
   else {
      assert(false);
   }

   double elapsedTime = timer.toc();
   std::cout << "done.\nOptimization took " << elapsedTime
             << " seconds.\n";

   // Print quality measures for the output mesh.
   geom::printQualityStatistics(std::cout, mesh);

   // Write the output mesh.
   writeAscii(parser.getArgument().c_str(), mesh);

   // Delete the boundary manifold if it was allocated.
   if (boundaryManifold != 0) {
      delete boundaryManifold;
      boundaryManifold = 0;
   }

   // There should be no more arguments.
   assert(parser.areArgumentsEmpty());

   return 0;
}
