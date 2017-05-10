// -*- C++ -*-

/*!
  \file nodesOnManifold.h
  \brief Get the indices of the nodes on the manifold.
*/

/*!
  \page examples_geom_mesh_node_on_manifold Get the indices of the nodes on the manifold.

  Determine which nodes in an N-D mesh are on an (N-1)-D manifold.
*/

#include "../iss_io.h"

#include "stlib/geom/mesh/iss/onManifold.h"
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
         << programName << " [-subset=file] [-epsilon=e] manifold mesh indices\n"
         << "  subset specifies a subset of the vertices.\n"
         << "  A node closer than epsilon to the manifold is on the manifold.\n"
         << "  manifold is the (N-1)-D manifold.\n"
         << "  mesh is the N-D simplicial mesh.\n"
         << "  indices is the output file.\n";

   exit(1);
}

}


//! The main loop.
int
main(int argc, char* argv[]) {
   typedef geom::IndSimpSet<SpaceDimension, SimplexDimension> Mesh;
   typedef geom::IndSimpSet < SpaceDimension, SimplexDimension - 1 > Manifold;

   ads::ParseOptionsArguments parser(argc, argv);

   programName = parser.getProgramName();

   //
   // Get the options.
   //

   // A subset of the nodes.
   bool areUsingSubset = false;
   std::string subsetFileName;
   std::vector<std::size_t> subset;
   if (parser.getOption("subset", &subsetFileName)) {
      areUsingSubset = true;
      std::ifstream file(subsetFileName.c_str());
      std::size_t size;
      file >> size;
      subset.resize(size);
      for (std::size_t i = 0; i != size; ++i) {
         file >> subset[i];
      }
   }

   // Distance for determining if a node is on the manifold.
   double epsilon = std::sqrt(std::numeric_limits<double>::epsilon());
   parser.getOption("epsilon", &epsilon);

   // Check that we parsed all of the options.
   if (! parser.areOptionsEmpty()) {
      std::cerr << "Error.  Unmatched options:\n";
      parser.printOptions(std::cerr);
      exitOnError();
   }

   // If they did not specify the manifold, the input mesh and output indices.
   if (parser.getNumberOfArguments() != 3) {
      std::cerr << "Bad number of required arguments.\n"
                << "You gave the arguments:\n";
      parser.printArguments(std::cerr);
      exitOnError();
   }

   // Read the manifold.
   Manifold manifold;
   readAscii(parser.getArgument().c_str(), &manifold);

   // Read the input mesh.
   Mesh mesh;
   readAscii(parser.getArgument().c_str(), &mesh);

   // Get the nodes of the manifold.
   std::vector<std::size_t> indices;
   if (areUsingSubset) {
      geom::determineVerticesOnManifold(mesh, subset.begin(), subset.end(),
                                        manifold,	std::back_inserter(indices),
                                        epsilon);
   }
   else {
      geom::determineVerticesOnManifold(mesh, manifold,
                                        std::back_inserter(indices), epsilon);
   }

   // Write the indices of the nodes on the manifold.
   {
      std::ofstream file(parser.getArgument().c_str());
      file << int(indices.size()) << "\n";
      for (std::vector<std::size_t>::const_iterator i = indices.begin();
            i != indices.end(); ++i) {
         file << *i << "\n";
      }
   }

   // There should be no more arguments.
   assert(parser.areArgumentsEmpty());

   return 0;
}
