// -*- C++ -*-

/*!
  \file pinballForces.cc
  \brief Report forces with the pinball method.
*/

/*!
  \page examples_geom_mesh_pinball_forces Pinball Contact

  \section examples_geom_mesh_pinball_forces_introduction Introduction

  This program takes a simplicial mesh and detects contact using the pinball
  method. It reports the forces to apply to the vertices.

  \section examples_geom_mesh_pinball_forces_usage Usage

  \verbatim
  pinballForcesN.exe mesh forces\endverbatim
  - mesh is the file name for the input mesh.
  - forces is the file name for recording the forces.
  .
  The forces are reported in the format:
  \verbatim
  numberOfForces
  vertexIdentifier forceX forceY forceZ
  ...\endverbatim

  \section examples_geom_mesh_pinball_forces_methodology Methodology

  See the \ref examples_geom_mesh_pinball "Pinball Contact" page for a
  description of how the contact between simplices is detected.

  We use a penalty method to calculate restoring forces that will remove
  contact between balls. For each contact pair, we use a spring force.
  The potential energy in the spring is equated to the kinetic energy
  of the balls, using only the components of the velocities that are normal
  to the contact plane.
*/

#ifndef DIMENSION
#error DIMENSION must be defined to compile this program.
#endif

#include "iss_io.h"

#include "stlib/geom/mesh/iss/pinballContact.h"
#include "stlib/ads/timer/Timer.h"
#include "stlib/ads/utility/ParseOptionsArguments.h"

#include <string>

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
         << " mesh forces\n\n"
         << "- mesh is the file name for the input mesh.\n"
         << "- forces is the file name for recording the forces.\n\n"
         << "The forces are reported in the format:\n"
         << "vertexIdentifier forceX forceY forceZ\n";
   exit(1);
}

}


//! The main loop.
int
main(int argc, char* argv[]) {
   static_assert(DIMENSION == 2 || DIMENSION == 3,
                 "The dimension must be 2 or 3.");

   // The simplicial mesh.
   typedef geom::IndSimpSet<DIMENSION, DIMENSION> Mesh;
   typedef std::array<double, DIMENSION> Point;
   typedef std::tuple<std::size_t, Point> Record;
   typedef std::array < std::size_t, DIMENSION + 1 > IndexedSimplex;

   ads::ParseOptionsArguments parser(argc, argv);
   programName = parser.getProgramName();

   // If they did not specify the mesh and output file.
   if (parser.getNumberOfArguments() != 2) {
      std::cerr << "Bad number of required arguments.\n"
                << "You gave the arguments:\n";
      parser.printArguments(std::cerr);
      exitOnError();
   }

   // Read the input solid mesh.
   Mesh mesh;
   readAscii(parser.getArgument().c_str(), &mesh);

   // Check that there are no options.
   if (! parser.areOptionsEmpty()) {
      std::cerr << "Error.  Unmatched options:\n";
      parser.printOptions(std::cerr);
      exitOnError();
   }

   // The inputs to pinballRestoringForces().
   std::vector<Point> vertices(mesh.getVerticesSize());
   for (std::size_t i = 0; i != vertices.size(); ++i) {
      vertices[i] = mesh.getVertex(i);
   }
   std::vector<std::size_t> vertexIdentifiers(mesh.getVerticesSize());
   for (std::size_t i = 0; i != vertexIdentifiers.size(); ++i) {
      vertexIdentifiers[i] = i;
   }
   std::vector<Point> velocities(mesh.getVerticesSize(),
                                 ext::filled_array<Point>(0.));
   std::vector<IndexedSimplex> identifierSimplices(mesh.getSimplicesSize());
   for (std::size_t i = 0; i != identifierSimplices.size(); ++i) {
      for (std::size_t m = 0; m != DIMENSION + 1; ++m) {
         identifierSimplices[i][m] = mesh.getIndexedSimplex(i)[m];
      }
   }
   std::vector<double> masses(mesh.getSimplicesSize(), 1.);
   const double maximumRelativePenetration = 0.1;
   std::vector<Record> forces;
   std::vector<std::size_t> interactionCounts;
   std::vector<double> relativePenetrations;

   // Compute the restoring forces.
   ads::Timer timer;
   timer.tic();
   geom::PinballRestoringForces<DIMENSION>
   pinballRestoringForces(maximumRelativePenetration);
   pinballRestoringForces(vertices, vertexIdentifiers, velocities,
                          identifierSimplices, masses,
                          std::back_inserter(forces),
                          std::back_inserter(interactionCounts),
                          std::back_inserter(relativePenetrations));
   double elapsedTime = timer.toc();

   {
      // Write the forces.
      std::ofstream out(parser.getArgument().c_str());
      out << forces.size() << '\n';
      for (std::vector<Record>::const_iterator i = forces.begin();
            i != forces.end(); ++i) {
         out << std::get<0>(*i) << ' ' << std::get<1>(*i) << '\n';
      }
   }

   // There should be no more arguments.
   assert(parser.areArgumentsEmpty());

   std::size_t maxInteractionCount = 0;
   if (! interactionCounts.empty()) {
      maxInteractionCount = *std::max_element(interactionCounts.begin(),
                                              interactionCounts.end());
   }
   double maxRelativePenetration = 0;
   if (! relativePenetrations.empty()) {
      maxRelativePenetration = *std::max_element(relativePenetrations.begin(),
                               relativePenetrations.end());
   }

   std::cout << "Computing forces took " << elapsedTime << " seconds.\n"
             << "Number of simplices = " << mesh.getSimplicesSize() << '\n'
             << "Number of forces = " << forces.size() << '\n'
             << "Maximum interaction count = " << maxInteractionCount << '\n'
             << "Maximum relative penetration = "
             << maxRelativePenetration << '\n'
             << "Time per simplex = "
             << elapsedTime / mesh.getSimplicesSize() * 1e6
             << " microseconds.\n";

   return 0;
}
