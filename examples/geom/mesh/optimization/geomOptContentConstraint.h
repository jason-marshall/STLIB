// -*- C++ -*-

/*!
  \file geomOptContentConstraint.h
  \brief Optimization of vertex locations with a content constraint.
*/

/*!
  \page examples_geom_mesh_geomOptContentConstraint Optimization of Simplicial Meshes with a Content Constraint.


  \section examples_geom_mesh_geomOptContentConstraint_examples Examples.

  \subsection mesh_smooth_examples_2D Optimize a 2-D Triangle Mesh.

  We will optimize the position of the interior and boundary vertices
  of a 2-D mesh.  The file \c ../data/square4.txt contains the triangle
  mesh of a square.

  \image html square4.gif "Triangle mesh of a square.  Mean condition number is 0.866."
  \image latex square4.pdf "Triangle mesh of a square.  Mean condition number is 0.866." width=0.5\textwidth

  Note that optimizing the interior vertices alone would have no effect.
  We take one optimization sweep with
  \verbatim
  geometricOptimize2 -boundary ../data/square4.txt square4_1.txt
  \endverbatim
  This moves only the boundary vertices.

  \image html square4_1.gif "One optimization sweep.  Mean condition number is 0.904."
  \image latex square4_1.pdf "One optimization sweep.  Mean condition number is 0.904." width=0.5\textwidth

  We take five optimization sweeps with
  \verbatim
  geometricOptimize2 -boundary -sweeps=5 ../data/square4.txt square4_5.txt
  \endverbatim
  Now all the vertices have been moved.

  \image html square4_5.gif "Five optimization sweeps.  Mean condition number is 1.035."
  \image latex square4_5.pdf "Five optimization sweeps.  Mean condition number is 1.035." width=0.5\textwidth

  After ten sweeps, the triangles are closer to equilateral.

  \image html square4_10.gif "Ten optimization sweeps.  Mean condition number is 0.986."
  \image latex square4_10.pdf "Ten optimization sweeps.  Mean condition number is 0.986." width=0.5\textwidth

  After this, there is diminishing return as the mesh slowly converges to
  its optimal shape.

  \image html square4_100.gif "One hundred optimization sweeps.  Mean condition number is 0.992."
  \image latex square4_100.pdf "One hundred optimization sweeps.  Mean condition number is 0.992." width=0.5\textwidth
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
         << " [-function=f] [-error=e]"
         << " [-sweeps=s] input output\n"
         << "- The function should be either 'm' for mean ratio or\n"
         << "  'c' for condition number.  By default it is mean ratio.\n"
         << "- error specifies the maximum allowed constraint error in the\n"
         << "  content (volume/area) of a simplex when moving a boundary vertex.\n"
         << "- The -sweeps option lets you specify the number of sweeps.\n"
         << "- input is the file name for the input mesh.\n"
         << "- output is the file name for the optimized mesh.\n";
   exit(1);
}

}


//! The main loop.
int
main(int argc, char* argv[]) {
   typedef geom::IndSimpSetIncAdj<Dimension, Dimension> Mesh;

   enum Function {MeanRatio, ConditionNumber};

   ads::ParseOptionsArguments parser(argc, argv);

   programName = parser.getProgramName();

   //
   // Get the options.
   //
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

   // The default maximum constraint error.
   double maximumConstraintError = 1.e-6;
   parser.getOption("error", &maximumConstraintError);

   // By default one sweep is performed.
   std::size_t numSweeps = 1;
   parser.getOption("sweeps", &numSweeps);
   if (numSweeps < 1) {
      std::cerr << "Bad number of sweeps.  You entered " << numSweeps << "\n";
      exitOnError();
   }

   // Check that we parsed all of the options.
   if (! parser.areOptionsEmpty()) {
      std::cerr << "Error.  Unmatched options:\n";
      parser.printOptions(std::cerr);
      exitOnError();
   }

   // If they did not specify the input mesh and output mesh.
   if (parser.getNumberOfArguments() != 2) {
      std::cerr << "Bad number of required arguments.\n"
                << "You gave the arguments:\n";
      parser.printArguments(std::cerr);
      exitOnError();
   }

   // Read the input mesh.
   Mesh mesh;
   readAscii(parser.getArgument().c_str(), &mesh);

   // Print quality measures for the input mesh.
   geom::printQualityStatistics(std::cout, mesh);

   std::cout << "Optimizing the mesh...\n" << std::flush;
   ads::Timer timer;
   timer.tic();

   // The set of interior vertices.
   std::vector<std::size_t> is;
   geom::determineInteriorVertices(mesh, std::back_inserter(is));

   // The set of boundary vertices.
   std::vector<std::size_t> bs;
   geom::determineComplementSetOfIndices(mesh.vertices.size(),
                                         is.begin(), is.end(),
                                         std::back_inserter(bs));
   if (function == MeanRatio) {
      for (std::size_t n = 0; n != numSweeps; ++n) {
         geom::geometricOptimizeUsingMeanRatio(&mesh, is.begin(), is.end());
         geom::geometricOptimizeConstrainedUsingMeanRatio
         (&mesh, bs.begin(), bs.end(), maximumConstraintError);
      }
   }
   else if (function == ConditionNumber) {
      for (std::size_t n = 0; n != numSweeps; ++n) {
         geom::geometricOptimizeUsingConditionNumber(&mesh, is.begin(), is.end());
         geom::geometricOptimizeConstrainedUsingConditionNumber
         (&mesh, bs.begin(), bs.end(), maximumConstraintError);
      }
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

   // There should be no more arguments.
   assert(parser.areArgumentsEmpty());

   return 0;
}
