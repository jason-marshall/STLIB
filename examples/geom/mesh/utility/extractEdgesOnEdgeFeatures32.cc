// -*- C++ -*-

/*!
  \file extractEdgesOnEdgeFeatures32.cc
  \brief Extract the edge features of a 3-2 manifold.
*/

/*!
  \page examples_geom_mesh_extractEdgesOnEdgeFeatures32 Extract the edge features of a 3-2 manifold.

  \section examples_geom_mesh_extractEdgesOnEdgeFeatures32_usage Usage.

  Command line options:
  \verbatim
  extractEdgesOnEdgeFeatures32 [-angle=maxDihedralAngleDeviation]
    [-featureDistance=d] manifold solidMesh edgeMesh
  \endverbatim
*/

#include "../iss_io.h"

#include "stlib/geom/mesh/iss/PointsOnManifold.h"
#include "stlib/ads/timer/Timer.h"
#include "stlib/ads/utility/ParseOptionsArguments.h"

#include <iostream>
#include <fstream>
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
         << programName
         << "\n [-angle=maxDihedralAngleDeviation] [-featureDistance=d]\n"
         << " manifold solidMesh edgeMesh\n"
         << "The angle specifies the maximum angle deviation for an edge feature.\n"
         << "By default, the angle is pi / 6.\n"
         << "The feature distance specifieds how far a point may be from an edge\n"
         << "and still be considered on the edge.\n"
         << "manifold defines the 3-2 manifold.\n"
         << "The boundary vertices of solidMesh (a 3-3 mesh) are registered\n"
         << "on the manifold.\n"
         << "edgeMesh is the output 3-1 mesh.\n";
   exit(1);
}

}


//! Main loop.
int
main(int argc, char* argv[]) {
   typedef geom::IndSimpSetIncAdj<3, 3> SolidMesh;
   typedef geom::IndSimpSet<3, 2> SurfaceMesh;
   typedef geom::IndSimpSet<3, 1> EdgeMesh;
   typedef geom::PointsOnManifold<3, 2, 1> Manifold;

   ads::ParseOptionsArguments parser(argc, argv);

   programName = parser.getProgramName();

   //
   // Get the options.
   //

   double maximumDihedralAngleDeviation =
      numerical::Constants<double>::Pi() / 6.;
   parser.getOption("angle", &maximumDihedralAngleDeviation);

   if (maximumDihedralAngleDeviation < 0) {
      std::cerr << "Bad maximum dihedral angle deviation.\n";
      exitOnError();
   }

   double featureDistance = -1;
   parser.getOption("featureDistance", &featureDistance);

   // Check that we parsed all of the options.
   if (! parser.areOptionsEmpty()) {
      std::cerr << "Error.  Unmatched options:\n";
      parser.printOptions(std::cerr);
      exitOnError();
   }

   // If they did not specify the two input and one output meshes.
   if (parser.getNumberOfArguments() != 3) {
      std::cerr << "Bad number of required arguments.\n"
                << "You gave the arguments:\n";
      parser.printArguments(std::cerr);
      exitOnError();
   }

   // Read the surface (manifold) mesh.
   SurfaceMesh surfaceMesh;
   readAscii(parser.getArgument().c_str(), &surfaceMesh);
   std::cout << "Quality of the surface mesh:\n";
   geom::printQualityStatistics(std::cout, surfaceMesh);
   std::cout << "\n";

   //
   // Make the manifold.
   //

   Manifold manifold(surfaceMesh, maximumDihedralAngleDeviation);
   // If the feature distance was specified.
   if (featureDistance >= 0) {
      manifold.setMaxCornerDistance(featureDistance);
      manifold.setMaxEdgeDistance(featureDistance);
   }

   // Read the solid mesh.
   SolidMesh solidMesh;
   readAscii(parser.getArgument().c_str(), &solidMesh);
   // CONTINUE
   std::cerr << "Register.\n";
   // Register the points and edges.
   manifold.insertBoundaryVerticesAndEdges(&solidMesh,
                                           maximumDihedralAngleDeviation);
   std::cerr << "Done.\n";

   // Print information about the manifold data structure.
   manifold.printInformation(std::cout);
   std::cout << "\n";

   // Get the edge mesh.
   EdgeMesh edgeMesh;
   manifold.getEdgesOnEdgeFeatures(solidMesh, &edgeMesh);
   // Pack to get rid of unused vertices.
   geom::pack(&edgeMesh);

   // Write the edge mesh.
   writeAscii(parser.getArgument().c_str(), edgeMesh);

   // There should be no more arguments.
   assert(parser.areArgumentsEmpty());

   return 0;
}
