// -*- C++ -*-

/*!
  \file topologicalOptimize.cc
  \brief Topological optimization via edge and face removal for tetrahedral meshes.
*/

/*!
  \page examples_geom_mesh_topologicalOptimize Topological Optimization for Tetrahedral Meshes

  \section examples_geom_mesh_topologicalOptimize_introduction Introduction.

  The file <tt>geom/mesh/tetrahedral/bin/topologicalOptimize.cc</tt>
  shows how to use the edge removal and face removal member functions of a
  geom::TetMesh.  It reads a mesh, optimizes the topology via edge removal
  and face removal and then writes the resulting mesh.


  \section examples_geom_mesh_topologicalOptimize_usage Usage.

  Command line options:
  \verbatim
  topologicalOptimize [-function=f] [-sweeps=n] [-steps=max]
    [-manifold=manifold] [-angle=maxDihedralAngleDeviation] input output
  \endverbatim
  The function should be either "m" for mean ratio or "c" for condition
  number.  By default it is mean ratio.  The -sweeps option lets you
  specify the maximum number of sweeps.  If the number of sweeps is not
  specified, the active tetrahedron will be stored and topological
  changes will be applied until the mesh cannot be further improved.
  If specified, the optimizations will take no more than the specified number
  of steps.  Edge removal of boundary edges will be applied only if the
  angle between the the normals of adjacent boundary faces is less
  than or equal to the specified angle.  By default the angle is 0.
  (The angle is in radians.)  \c output is the file name for the
  optimized mesh.

  Make a single sweep over all the edges and interior faces of a tetrahedral
  mesh using the mean ratio metric with:
  \verbatim
  topologicalOptimize -sweeps=1 mesh.txt opt_mesh.txt
  \endverbatim
  Fully optimize the mesh using the condition number metric with:
  \verbatim
  topologicalOptimize -function=c mesh.txt opt_mesh.txt
  \endverbatim

  Consult the documentation for geom::TetMesh for the file format.
  This program reads and writes meshes in ascii format.

  \section examples_geom_mesh_topologicalOptimize_examples Examples.

  We compare the four methods of optimizing the deformed cylinder.  For
  the geometric optimization, we performed one sweep over all interior
  nodes.  Below we list the optimization method and the resulting
  minimum condition number.
  - Original: 0.15
  - Geometric: 0.20
  - Topological: 0.28
  - Geometric then topological: 0.29
  - Topological then geometric: 0.36

  \image html cylinder.jpg "A deformed cylinder."
  \image latex cylinder.pdf "A deformed cylinder." width=\textwidth
*/

#include "../smr_io.h"
#include "../iss_io.h"

#include "stlib/geom/mesh/simplicial/topologicalOptimize.h"
#include "stlib/geom/mesh/simplicial/quality.h"
#include "stlib/geom/mesh/iss/build.h"
#include "stlib/ads/timer/Timer.h"
#include "stlib/ads/utility/ParseOptionsArguments.h"

#include <iostream>
#include <fstream>
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
         << programName << " [-function=f] [-sweeps=n] [-steps=max]\n"
         << "[-manifold=manifold] [-angle=maxDihedralAngleDeviation] input output\n"
         << "The function should be either 'm' for mean ratio or\n"
         << "'c' for condition number.  By default it is mean ratio.\n"
         << "The -sweeps option lets you specify the number of sweeps.\n"
         << "If specified, the optimization will take no more than "
         << "the maximum indicated steps.\n"
         << "manifold specifies the boundary manifold.\n"
         << "angle specifies the maximum boundary angle deviation for edge removal.\n"
         << "input is the file name for the input mesh.\n"
         << "output is the file name for the optimized mesh.\n";
   exit(1);
}

}


//! Main loop.
int
main(int argc, char* argv[]) {
   typedef geom::SimpMeshRed<3, 3> Mesh;
   typedef geom::IndSimpSetIncAdj<3, 3> ISS;
   typedef geom::IndSimpSet<3, 2> BoundaryMesh;
   typedef geom::PointsOnManifold<3, 2, 1> BoundaryManifold;

   enum Function {MeanRatio, ConditionNumber};

   ads::ParseOptionsArguments parser(argc, argv);

   programName = parser.getProgramName();

   //
   // Get the options.
   //
   // The default function is mean ratio.
   Function function = MeanRatio;
   {
      // Initialize with a bad option.
      char f = 'a';
      // If they specifed a function.
      if (parser.getOption("function", &f)) {
         if (f == 'm') {
            function = MeanRatio;
         }
         else if (f == 'c') {
            function = ConditionNumber;
         }
         else {
            std::cerr << "Bad function.  Exiting.\n";
            exitOnError();
         }
      }
   }

   // By default we sweep until no further optimizations can be performed.
   int numberOfSweeps = 0;
   parser.getOption("sweeps", &numberOfSweeps);

   std::size_t maximumNumberOfSteps = std::numeric_limits<std::size_t>::max();
   parser.getOption("steps", &maximumNumberOfSteps);

   double featureDistance = 0;
   if (parser.getOption("featureDistance", &featureDistance)) {
      if (featureDistance <= 0) {
         std::cerr << "Bad feature distance.\n";
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

   //
   // Read the boundary mesh.
   //

   // Get the maximum dihedral angle deviation for a surface feauture.
   double maxDihedralAngleDeviation = -1;
   parser.getOption("angle", &maxDihedralAngleDeviation);

   std::string boundaryFile;
   BoundaryManifold* boundaryManifold = 0;
   BoundaryMesh boundaryMesh;
   // If they specified a file for the boundary manifold.
   if (parser.getOption("manifold", &boundaryFile)) {
      readAscii(boundaryFile.c_str(), &boundaryMesh);
   }
   // Else, extract the boundary of the input mesh.
   else {
      ISS iss;
      geom::buildIndSimpSetFromSimpMeshRed(mesh, &iss);
      geom::buildBoundary(iss, &boundaryMesh);
   }
   std::cout << "Quality of the boundary mesh:\n";
   geom::printQualityStatistics(std::cout, boundaryMesh);
   std::cout << "\n";
   boundaryManifold = new BoundaryManifold(boundaryMesh,
                                           maxDihedralAngleDeviation);

   // If the feature distance was specified.
   if (featureDistance != 0) {
      // Set the maximum corner distance.
      boundaryManifold->setMaxCornerDistance(featureDistance);
      // Set the maximum edge distance.
      boundaryManifold->setMaxEdgeDistance(featureDistance);
   }
   // Register points and edges of the boundary.
   boundaryManifold->insertBoundaryVerticesAndEdges
   (&mesh, maxDihedralAngleDeviation);

   // Print information about the manifold data structure.
   if (boundaryManifold != 0) {
      boundaryManifold->printInformation(std::cout);
      std::cout << "\n";
   }

   // Check that we parsed all of the options.
   if (! parser.areOptionsEmpty()) {
      std::cerr << "Error.  Unmatched options:\n";
      parser.printOptions(std::cerr);
      exitOnError();
   }

   // Print quality measures for the input mesh.
   std::cout << "Quality of the input mesh:\n";
   geom::printQualityStatistics(std::cout, mesh);
   std::cout << "\n";

   if (function == MeanRatio) {
      std::cout << "Optimizing the mesh using the mean ratio metric...\n"
                << std::flush;
   }
   else if (function == ConditionNumber) {
      std::cout << "Optimizing the mesh using the condition number metric...\n"
                << std::flush;
   }
   else {
      assert(false);
   }

   ads::Timer timer;
   timer.tic();

   std::multiset<std::pair<std::size_t, std::size_t> > edgeRemovalOperations;
   std::multiset<std::pair<std::size_t, std::size_t> > faceRemovalOperations;
   if (function == MeanRatio) {
      std::cout
            << geom::topologicalOptimizeUsingMeanRatio(&mesh, boundaryManifold,
                  &edgeRemovalOperations,
                  &faceRemovalOperations,
                  maximumNumberOfSteps)
            << " edge and face removals.\n";
   }
   else if (function == ConditionNumber) {
      std::cout
            << geom::topologicalOptimizeUsingConditionNumber(&mesh, boundaryManifold,
                  &edgeRemovalOperations,
                  &faceRemovalOperations,
                  maximumNumberOfSteps)
            << " edge and face removals.\n";
   }
   else {
      assert(false);
   }

   // CONTINUE
#if 0
   if (numberOfSweeps == 0) {
      if (function == MeanRatio) {
         std::cout << mesh.optimizeUsingMeanRatio(maximumNumberOfSteps)
                   << " edge and face removals.\n";
      }
      else if (function == ConditionNumber) {
         std::cout << mesh.optimizeUsingConditionNumber(maximumNumberOfSteps)
                   << " edge and face removals.\n";
      }
      else {
         assert(false);
      }
   }
   else {
      if (function == MeanRatio) {
         for (std::size_t n = 0; n != numberOfSweeps; ++n) {
            std::cout << mesh.applyEdgeRemovalSweepUsingMeanRatio
                      (maximumNumberOfSteps)
                      << " edge removals.\n";
            std::cout << mesh.applyFaceRemovalSweepUsingMeanRatio
                      (maximumNumberOfSteps)
                      << " face removals.\n";
         }
      }
      else if (function == ConditionNumber) {
         for (std::size_t n = 0; n != numberOfSweeps; ++n) {
            std::cout << mesh.applyEdgeRemovalSweepUsingConditionNumber
                      (maximumNumberOfSteps)
                      << " edge removals.\n";
            std::cout << mesh.applyFaceRemovalSweepUsingConditionNumber
                      (maximumNumberOfSteps)
                      << " face removals.\n";
         }
      }
      else {
         assert(false);
      }
   }
#endif

   double elapsedTime = timer.toc();
   std::cout << "done.\nOptimization took " << elapsedTime
             << " seconds.\n";

   // CONTINUE
   // Print operation statistics for the optimization.
   //mesh.printOperationStatistics(std::cout);

   // Print quality measures for the output mesh.
   geom::printQualityStatistics(std::cout, mesh);

   // CONTINUE
   //assert(mesh.isValid());

   // Write the output mesh.
   writeAscii(parser.getArgument().c_str(), mesh);

   // There should be no more arguments.
   assert(parser.areArgumentsEmpty());

   // If a boundary manifold was allocated.
   if (boundaryManifold) {
      // Free that memory.
      delete boundaryManifold;
      boundaryManifold = 0;
   }

   return 0;
}
