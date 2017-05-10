// -*- C++ -*-

// CONTINUE: Change this back to using the old method.  Then I will have
// something with which to compare the PointsOnManifold approach.
/*!
  \file geomOptBoundaryCondition.h
  \brief Optimization of vertex locations with a boundary constraint.
*/

/*!
  \page examples_geom_mesh_geomOptBoundaryCondition Optimization of Simplicial Meshes with a Boundary Constraint.



  \section examples_geom_mesh_geomOptBoundaryCondition_introduction Introduction

  This program uses geometric optimization in conjuction with a boundary
  condition.  To move the boundary vertices, the vertex position is optimized
  unconstrained or subject to a constant content constraint.  Then the
  position is projected onto the boundary curve/surface.  If the quality
  of the mesh is improved by this change, it is accepted.



  \section examples_geom_mesh_geomOptBoundaryCondition_example2 2-D Example

  We start with the boundary of the unit square.

  \verbatim
  cp ../../../data/geom/mesh/21/square.txt boundary.txt
  \endverbatim

  \image html geomOptBoundaryCondition_2_boundary.jpg "The boundary curve."
  \image latex geomOptBoundaryCondition_2_boundary.pdf "The boundary curve."

  We get an initial mesh that does not match the curve at the boundary.

  \verbatim
  cp ../../../data/geom/mesh/22/diamond.txt diamond.txt
  \endverbatim

  \image html geomOptBoundaryCondition_2_diamond.jpg "The diamond does not match the boundary very well."
  \image latex geomOptBoundaryCondition_2_diamond.pdf "The diamond does not match the boundary very well."

  Next we apply refinement to the diamond mesh until no edge length exceeds
  0.1.  Each triangle in this mesh has a condition number of 0.87.

  \verbatim
  refine2.exe -length=0.1 diamond.txt refined.txt
  \endverbatim

  \image html geomOptBoundaryCondition_2_refined.jpg "The refined diamond mesh."
  \image latex geomOptBoundaryCondition_2_refined.pdf "The refined diamond mesh."

  we move the boundary vertices of the refined mesh to the boundary curve.
  This produces triangles with low quality.  The minimum condition
  number is 0.079; the mean is 0.73.

  \verbatim
  moveBoundary2.exe boundary.txt refined.txt mesh.txt
  \endverbatim

  \image html geomOptBoundaryCondition_2_mesh.jpg "A poor quality mesh for the unit square."
  \image latex geomOptBoundaryCondition_2_mesh.pdf "A poor quality mesh for the unit square."

  Now we apply geometric optimization with a boundary condition.  After
  one sweep, the minimum condition number is 0.27; the mean is 0.76.

  \verbatim
  geomOptBoundaryCondition2.exe boundary.txt mesh.txt mesh_1.txt
  \endverbatim

  \image html geomOptBoundaryCondition_2_mesh_1.jpg "One sweep of geometric optimization."
  \image latex geomOptBoundaryCondition_2_mesh_1.pdf "One sweep of geometric optimization."

  More sweeps produce better quality meshes.
  After two optimization sweeps,
  the minimum condition number is 0.42, the mean is 0.79.

  \verbatim
  geomOptBoundaryCondition2.exe boundary.txt mesh_1.txt mesh_2.txt
  \endverbatim

  \image html geomOptBoundaryCondition_2_mesh_2.jpg "Two sweeps of geometric optimization."
  \image latex geomOptBoundaryCondition_2_mesh_2.pdf "Two sweeps of geometric optimization."

  After ten optimization sweeps,
  the minimum condition number is 0.57, the mean is 0.83.  This is by no means
  a high quality mesh of the unit square, but it is a large improvement
  over the initial mesh.

  \verbatim
  geomOptBoundaryCondition2.exe -sweeps=8 boundary.txt mesh_2.txt mesh_10.txt
  \endverbatim

  \image html geomOptBoundaryCondition_2_mesh_10.jpg "Ten sweeps of geometric optimization."
  \image latex geomOptBoundaryCondition_2_mesh_10.pdf "Ten sweeps of geometric optimization."

  Looking at the above mesh, it is apparent that we have taken a mesh for
  the diamond and stretched it to try to fit the square.  Inevitably, the
  result is not very satisfying.
  To obtain a better quality triangles, we would
  have to optimize the topology.  To obtain better agreement with the corners
  of the boundary, we would have to refine the mesh there.

  We'll try the topological optimization first.
  Apter applying edge flipping and smoothing,
  the minimum condition number is 0.75, the mean is 0.93.

  \verbatim
  flip2.exe mesh_10.txt mesh_gf.txt
  geomOptBoundaryCondition2.exe -sweeps=2 boundary.txt mesh_gf.txt mesh_gfg.txt
  \endverbatim

  \image html geomOptBoundaryCondition_2_mesh_gfg.jpg "Topological and geometric optimization."
  \image latex geomOptBoundaryCondition_2_mesh_gfg.pdf "Topological and geometric optimization."


  Next we'll bring in refinement to address the problem of matching the
  boundary.  We apply on cycle of refinement, topological optimization, and
  geometric optimization.  (Following the refinement we move the
  new boundary vertices to the boundary curve.)

  For the resulting mesh, the minimum condition number is 0.70,
  the mean is 0.94.

  \verbatim
  refine2.exe -length=0.1 mesh_gfg.txt mesh_gfgc.txt
  moveBoundary2.exe boundary.txt mesh_gfgc.txt mesh_gfgc.txt
  flip2.exe mesh_gfgc.txt mesh_gfgc.txt
  geomOptBoundaryCondition2.exe -sweeps=2 boundary.txt mesh_gfgc.txt mesh_gfgc.txt
  \endverbatim

  \image html geomOptBoundaryCondition_2_mesh_gfgc.jpg "One cycle of refinement, topological optimization and geometric optimization."
  \image latex geomOptBoundaryCondition_2_mesh_gfgc.pdf "One cycle of refinement, topological optimization and geometric optimization."


  Now we are getting somewhere.  The above mesh has good quality triangles
  and roughly matches the boundary curve.  If we apply one more cycle of
  refinement, topological optimization, and geometric optimization, we
  will further improve the situation at the corners.

  For this final mesh, the minimum condition number is 0.76,
  the mean is 0.95.

  \verbatim
  refine2.exe -length=0.1 mesh_gfgc.txt mesh_gfgcc.txt
  moveBoundary2.exe boundary.txt mesh_gfgcc.txt mesh_gfgcc.txt
  flip2.exe mesh_gfgcc.txt mesh_gfgcc.txt
  geomOptBoundaryCondition2.exe -sweeps=2 boundary.txt mesh_gfgcc.txt mesh_gfgcc.txt
  \endverbatim

  \image html geomOptBoundaryCondition_2_mesh_gfgcc.jpg "Two cycles of refinement, topological optimization and geometric optimization."
  \image latex geomOptBoundaryCondition_2_mesh_gfgcc.pdf "Two cycles of refinement, topological optimization and geometric optimization."
*/

#include "../iss_io.h"

#include "stlib/geom/mesh/iss/optimize.h"
#include "stlib/geom/mesh/iss/set.h"
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
//! The default function is mean ratio.
static Function function = MeanRatio;
//! By default one sweep is performed.
static std::size_t numSweeps = 1;
//! By default perform one interior optimization cycle per sweep.
static std::size_t numInterior = 1;
//! By default perform one boundary optimization cycle per sweep.
static std::size_t numBoundary = 1;
#if DIMENSION == 2
//! The maximum angle deviation for a surface feature.
static double maxAngleDeviation = -1;
#elif DIMENSION == 3
//! The maximum dihedral angle deviation for a surface feature.
static double maxDihedralAngleDeviation = -1;
//! The maximum solid angle deviation for a surface feature.
static double maxSolidAngleDeviation = -1;
//! The maximum boundary angle deviation for a boundary edge feature.
static double maxBoundaryAngleDeviation = -1;
#endif

// Local functions.

// Not used.
#if 0
//! Smooth the mesh.
void
geometricOptimize(Mesh* mesh, BoundaryManifold* boundaryManifold) {
   // The set of interior vertices.
   std::vector<std::size_t> is;
   geom::determineInteriorVertices(*mesh, std::back_inserter(is));

   // The set of boundary vertices.
   std::vector<std::size_t> bs;
   geom::determineComplementSetOfIndices(mesh->vertices.size(),
                                         is.begin(), is.end(),
                                         std::back_inserter(bs));
   if (function == MeanRatio) {
      for (std::size_t n = 0; n != numSweeps; ++n) {
         geom::geometricOptimizeUsingMeanRatio(mesh, is.begin(), is.end(),
                                               numInterior);
         geom::geometricOptimizeWithBoundaryConditionUsingMeanRatio
         (mesh, bs.begin(), bs.end(), boundaryManifold, numBoundary);
      }
   }
   else if (function == ConditionNumber) {
      for (std::size_t n = 0; n != numSweeps; ++n) {
         geom::geometricOptimizeUsingConditionNumber(mesh, is.begin(), is.end(),
               numInterior);
         geom::geometricOptimizeWithBoundaryConditionUsingConditionNumber
         (mesh, bs.begin(), bs.end(), boundaryManifold, numBoundary);
      }
   }
   else {
      assert(false);
   }
}
#endif

//! Exit with an error message.
void
exitOnError() {
   std::cerr
         << "Bad arguments.  Usage:\n"
         << programName << "\n"
         << " [-function=f] "
         << " [-sweeps=n] [-interior=subcycles] [-boundary=subcycles]\n"
#if DIMENSION == 2
         << " [-angle=a]\n"
#elif DIMENSION == 3
         << " [-dihedralAngle=d] [-solidAngle=s]\n"
         << " [-boundaryAngle=b]\n"
#endif
         << " boundary input output\n\n"
         << "- The function should be either 'm' for mean ratio or\n"
         << "  'c' for condition number.  By default it is mean ratio.\n"
         << "- The -sweeps option lets you specify the number of sweeps.\n"
         << "- The -interior option lets you specify the number of interior optimization\n"
         << "  cycles per sweep.\n"
         << "- The -boundary option lets you specify the number of boundary optimization\n"
         << "  cycles per sweep.\n"
#if DIMENSION == 2
         << "- angle is the maximum angle deviation (from pi)\n"
#elif DIMENSION == 3
         << "- dihedralAngle is the maximum dihedral angle deviation\n"
         << "  (from straight) for a surface feature.  The rest are edge "
         << "features.\n"
         << "- Solid angles that deviate more than maxSolidAngleDeviation\n"
         << "  (from 2*pi) are corner features.\n"
         << "- If the angle deviation (from pi) between two boundary edges \n"
         << "  exceeds maxBoundaryAngleDeviation, it will be set as a corner "
         << "feature.\n"
         << "  between two boundary edges for a corner feature on the boundary.\n"
#endif
         << "- boundary is the file name for the boundary mesh.\n"
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

   parser.getOption("sweeps", &numSweeps);
   if (numSweeps < 1) {
      std::cerr << "Bad number of sweeps.  You entered " << numSweeps << "\n";
      exitOnError();
   }

   parser.getOption("boundary", &numBoundary);

   parser.getOption("interior", &numInterior);

#if DIMENSION == 2
   parser.getOption("angle", &maxAngleDeviation);
#elif DIMENSION == 3
   parser.getOption("dihedralAngle", &maxDihedralAngleDeviation);
   parser.getOption("solidAngle", &maxSolidAngleDeviation);
   parser.getOption("boundaryAngle", &maxBoundaryAngleDeviation);
#endif

   // Check that we parsed all of the options.
   if (! parser.areOptionsEmpty()) {
      std::cerr << "Error.  Unmatched options:\n";
      parser.printOptions(std::cerr);
      exitOnError();
   }

   // If they did not specify the boundary, input mesh, and output mesh.
   if (parser.getNumberOfArguments() != 3) {
      std::cerr << "Bad number of required arguments.\n"
                << "You gave the arguments:\n";
      parser.printArguments(std::cerr);
      exitOnError();
   }

   // Read the input boundary mesh.
   Boundary boundary;
   readAscii(parser.getArgument().c_str(), &boundary);
   std::cout << "The boundary mesh has " << boundary.vertices.size()
             << " vertices and " << boundary.indexedSimplices.size()
             << " simplices.\n";

   // Read the input solid mesh.
   Mesh mesh;
   readAscii(parser.getArgument().c_str(), &mesh);
   std::cout << "The solid mesh has " << mesh.vertices.size()
             << " vertices and " << mesh.indexedSimplices.size()
             << " simplices.\n";

   // The boundary manifold.
#if DIMENSION == 2
   BoundaryManifold boundaryManifold(boundary, maxAngleDeviation);
   // Register points at each of the boundary vertices.
   boundaryManifold.insertBoundaryVertices(&mesh);
#elif DIMENSION == 3
   BoundaryManifold boundaryManifold(boundary, maxDihedralAngleDeviation,
                                     maxSolidAngleDeviation,
                                     maxBoundaryAngleDeviation);
   // Register points and edges of the boundary.
   boundaryManifold.insertBoundaryVerticesAndEdges(&mesh);
#endif

   // Print quality measures for the input mesh.
   geom::printQualityStatistics(std::cout, mesh);

   std::cout << "Optimizing the mesh...\n" << std::flush;
   ads::Timer timer;
   timer.tic();

   // Geometric optimization.
   geometricOptimizeUsingMeanRatio(&mesh, &boundaryManifold);

   double elapsedTime = timer.toc();
   std::cout << "done.\nOptimization took " << elapsedTime
             << " seconds.\n";

   // Print quality measures for the output mesh.
   geom::printQualityStatistics(std::cout, mesh);

   // Write the output mesh.
   writeAscii(parser.getArgument().c_str(), mesh);

   // There should be no more arguments.
   assert(parser.areArgumentsEmpty());

   return 0;
}
