// -*- C++ -*-

/*!
  \file flip.h
  \brief Flip edges to improve the quality of the mesh.
*/

/*!
  \page examples_geom_mesh_flip Flip edges in a 2-D mesh.

  \section examples_geom_mesh_flip_example Example

  We start with a coarse initial mesh that has fairly poor quality.
  This mesh has 10 vertices and 10 triangles.  The minimum condition
  number is 0.055; the mean is 0.25.

  \verbatim
  cp ../../../data/geom/mesh/22/a.txt initial.txt
  \endverbatim

  \image html flip_2_initial.jpg "The initial coarse mesh."
  \image latex flip_2_initial.pdf "The initial coarse mesh."

  We refine the mesh so no edge is longer than 0.1.  This results in a mesh
  that has plenty of poor quality triangles.
  The refined mesh has 154 vertices and 241 triangles.  The minimum condition
  number is 0.055; the mean is 0.56.

  \verbatim
  refine2 -length=0.1 initial.txt refined.txt
  \endverbatim

  \image html flip_2_refined.jpg "The refined mesh."
  \image latex flip_2_refined.pdf "The refined mesh."

  Then we apply edge flipping to improve the quality of the mesh.
  61 edges are flipped.
  The minimum condition number is 0.28; the mean is 0.66.

  \verbatim
  flip22 refined.txt flipped.txt
  \endverbatim

  \image html flip_2_flipped.jpg "The mesh after edge flips."
  \image latex flip_2_flipped.pdf "The mesh after edge flips."
*/

#include "../smr_io.h"

#include "stlib/geom/mesh/simplicial/flip.h"
#include "stlib/geom/mesh/simplicial/quality.h"
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
         << programName << " [-angle=maximumAngle] [-function=f] in out\n"
         << "-angle is used to specify the maximum allowed angle between face.\n"
         << "  normals for flippable edges.  This is an option for 3-2\n"
         << "  meshes only.\n"
         << "-function is used to specify the simplex quality function.\n"
         << "  The function should be either 'm' for mean ratio or\n"
         << "  'c' for condition number.  By default it is mean ratio.\n"
         << "in is the input mesh.\n"
         << "out is the output mesh.\n";
   exit(1);
}

}


//! The main loop.
int
main(int argc, char* argv[]) {
   static_assert(SpaceDimension == 2 || SpaceDimension == 3,
                 "The space dimension must be 2 or 3.");
   static_assert(SimplexDimension == 2, "The simplex dimension must be 2.");

   typedef geom::SimpMeshRed<SpaceDimension, SimplexDimension> Mesh;

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

#if (SpaceDimension == 3)
   // The default value of the maximum angle is pi/18.
   double maximumAngle = numerical::Constants<double>::Pi() / 18.0;
   parser.getOption("angle", &maximumAngle);
   if (maximumAngle < 0) {
      std::cerr << "Bad angle.  You specified " << maximumAngle << "\n";
      exitOnError();
   }
#endif

   // Check that we parsed all of the options.
   if (! parser.areOptionsEmpty()) {
      std::cerr << "Error.  Unmatched options:\n";
      parser.printOptions(std::cerr);
      exitOnError();
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

   // Print quality measures for the input mesh.
   std::cout << "Quality of the input mesh:\n";
   geom::printQualityStatistics(std::cout, mesh);

   std::cout << "\nFlipping edges...\n" << std::flush;
   ads::Timer timer;
   timer.tic();

   int count;
   if (function == MeanRatio) {
#if (SpaceDimension == 3)
      count = geom::flipUsingModifiedMeanRatio(&mesh, maximumAngle);
#else
      count = geom::flipUsingModifiedMeanRatio(&mesh);
#endif
   }
   else if (function == ConditionNumber) {
#if (SpaceDimension == 3)
      count = geom::flipUsingModifiedConditionNumber(&mesh, maximumAngle);
#else
      count = geom::flipUsingModifiedConditionNumber(&mesh);
#endif
   }
   else {
      assert(false);
   }

   double elapsedTime = timer.toc();
   std::cout << "done.\n"
             << "Number of edges flipped = " << count << "\n"
             << "Flipping took " << elapsedTime << " seconds.\n";

   // Print quality measures for the output mesh.
   std::cout << "\nQuality of the output mesh:\n";
   geom::printQualityStatistics(std::cout, mesh);

   // Write the output mesh.
   writeAscii(parser.getArgument().c_str(), mesh);

   // There should be no more arguments.
   assert(parser.areArgumentsEmpty());

   return 0;
}
