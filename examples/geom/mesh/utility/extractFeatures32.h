// -*- C++ -*-

/*!
  \file extractFeatures32.h
  \brief Extract the edge features of a 3-2 manifold.
*/

/*!
  \page examples_geom_mesh_extractFeatures32 Extract the edge and corner features of a 3-2 manifold.

  \section examples_geom_mesh_extractFeatures32_usage Usage.

  Command line options:
  \verbatim
  extractFeatures32 [-dihedralAngle=d] [-solidAngle=s] [-boundaryAngle=b]
    inputMesh outputEdges outputCorners
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
         << " [-dihedralAngle=d] [-solidAngle=s] [-boundaryAngle=b]\n"
         << " inputMesh outputEdges outputCorners\n"
         << "- The dihedral angle specifies the maximum angle deviation for\n"
         << "  an edge feature.\n"
         << "- The solid angle is used to identify corner features.\n"
         << "- The boundary angle is used to identify corner features on the boundary.\n"
         << "inputMesh is the input surface (3-2) mesh.\n"
         << "outputEdges is the output edge (3-1) mesh.\n"
         << "outputCorners is the output corner (3-0) mesh.\n";
   exit(1);
}

}


//! Main loop.
int
main(int argc, char* argv[]) {
   typedef geom::IndSimpSet<3, 2> Mesh;
   typedef geom::IndSimpSet<3, 1> EdgeMesh;
   typedef geom::IndSimpSet<3, 0> CornerMesh;
   typedef geom::PointsOnManifold<3, 2, 1> Manifold;

   ads::ParseOptionsArguments parser(argc, argv);

   programName = parser.getProgramName();

   //
   // Get the options.
   //
   double maximumDihedralAngleDeviation = -1;
   parser.getOption("dihedralAngle", &maximumDihedralAngleDeviation);

   double maximumSolidAngleDeviation = -1;
   parser.getOption("solidAngle", &maximumSolidAngleDeviation);

   double maximumBoundaryAngleDeviation = -1;
   parser.getOption("boundaryAngle", &maximumBoundaryAngleDeviation);

   // Check that we parsed all of the options.
   if (! parser.areOptionsEmpty()) {
      std::cerr << "Error.  Unmatched options:\n";
      parser.printOptions(std::cerr);
      exitOnError();
   }

   // If they did not specify the input mesh and the two output meshes.
   if (parser.getNumberOfArguments() != 3) {
      std::cerr << "Bad number of required arguments.\n"
                << "You gave the arguments:\n";
      parser.printArguments(std::cerr);
      exitOnError();
   }

   // Read the input mesh.
   Mesh mesh;
   readAscii(parser.getArgument().c_str(), &mesh);
   std::cout << "Quality of the surface mesh:\n";
   geom::printQualityStatistics(std::cout, mesh);
   std::cout << "\n";

   //
   // Make the manifold.
   //

   Manifold manifold(mesh, maximumDihedralAngleDeviation,
                     maximumSolidAngleDeviation,
                     maximumBoundaryAngleDeviation);
   // Print information about the manifold data structure.
   manifold.printInformation(std::cout);
   std::cout << "\n";

   // Get the edge mesh.
   EdgeMesh edgeMesh(manifold.getEdgeManifold());
   // Pack to get rid of unused vertices.
   geom::pack(&edgeMesh);
   // Write the edge mesh.
   writeAscii(parser.getArgument().c_str(), edgeMesh);

   // Get the corner mesh.
   CornerMesh cornerMesh(mesh.getVerticesSize(), &mesh.getVertices()[0][0],
                         manifold.getCornerIndices().size(),
                         &manifold.getCornerIndices()[0]);
   // Pack to get rid of unused vertices.
   geom::pack(&cornerMesh);
   // Write the edge mesh.
   writeAscii(parser.getArgument().c_str(), cornerMesh);

   // There should be no more arguments.
   assert(parser.areArgumentsEmpty());

   return 0;
}
