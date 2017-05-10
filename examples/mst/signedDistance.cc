// -*- C++ -*-

/*!
  \file signedDistance.cc
  \brief Compute the signed distance to the molecular surface for a set of points.
*/

/*!
\page mst_signedDistance Signed Distance to the Molecular Surface.

<!--------------------------------------------------------------------------->
\section mstSignedDistanceIntroduction Introduction.

This program reads a molecule in xyzr format, and a set of points.
It computes the signed distance to the molecular surface for each of
the points and writes the result as an attribute file.

<!--------------------------------------------------------------------------->
\section mst_signedDistanceUsage Usage.

\verbatim
signedDistance.exe [-radius=r] molecule.xyzr points.txt attribute.txt
\endverbatim

- radius specifies the probe radius.  By default it is 0.
- molecule.xyzr is the input file containing the atom's centers and radii.
- points.txt is the input set of points.
- attribute.txt is the output signed distance.

The molecule input file simply contains an enumeration of the centers and
radii of the atoms in the molecule.  For example, a molecule with a single atom
of radius 1 Angstrom and centered at the origin would have the file:
\verbatim
0.0 0.0 0.0 1.0
\endverbatim

The input points have the format:
\verbatim
numberOfPoints
x0 y0 z0
x1 y1 z1
...
\endverbatim

The output file has the format:
\verbatim
Distance
numberOfPoints
distance0
distance1
...
\endverbatim
The first line gives the name of the attribute.


<!--------------------------------------------------------------------------->
\section mst_signedDistanceExample Example.

CONTINUE.
*/

#include "stlib/mst/readXyzr.h"
#include "stlib/mst/Molecule.h"

#include "stlib/ads/timer/Timer.h"
#include "stlib/ads/utility/ParseOptionsArguments.h"
#include "stlib/ext/vector.h"

#include <sstream>

#include <cassert>

USING_STLIB_EXT_VECTOR_IO_OPERATORS;
using namespace stlib;

namespace {

//
// Global variables.
//

//! The program name.
std::string programName;

//
// Local functions.
//

//! Exit with an error message.
void
exitOnError() {
   std::cerr
         << "Bad arguments.  Usage:\n"
         << programName
         << " [-radius=r] molecule.xyzr points.txt attribute.txt\n"
         << "- radius specifies the probe radius.  By default it is 0.\n"
         << "- molecule.xyzr is the input file containing the atom's centers and radii.\n"
         << "- points.txt is the input set of points.\n"
         << "- attribute.txt is the output signed distance.\n";
   exit(1);
}

}


//! The main loop.
int
main(int argc, char* argv[]) {
   typedef double NumberType;
   typedef mst::Molecule<NumberType> Molecule;
   typedef Molecule::AtomType AtomType;
   typedef Molecule::Point Point;

   ads::ParseOptionsArguments parser(argc, argv);

   // Program name.
   programName = parser.getProgramName();

   // If they did not specify the two input and one output files.
   if (parser.getNumberOfArguments() != 3) {
      std::cerr << "Bad number of required arguments.\n"
                << "You gave the arguments:\n";
      parser.printArguments(std::cerr);
      exitOnError();
   }

   // Probe radius.
   NumberType probeRadius = 0;
   parser.getOption("radius", &probeRadius);

   // Check that we parsed all of the options.
   if (! parser.areOptionsEmpty()) {
      std::cerr << "Error.  Unmatched options:\n";
      parser.printOptions(std::cerr);
      exitOnError();
   }

   // The atoms.
   std::vector<AtomType> atoms;

   // Read the input file.
   std::cout << "Reading the input file...\n" << std::flush;
   mst::readXyzr<NumberType>(parser.getArgument().c_str(),
                             std::back_inserter(atoms));

   // Offset by the probe radius.
   if (probeRadius != 0) {
      for (std::vector<AtomType>::iterator i = atoms.begin(); i != atoms.end();
            ++i) {
         // Make sure we don't have an atom with negative radius.
         if (i->radius + probeRadius <= 0) {
            std::cerr << "Bad value for the probe radius.\n";
            exitOnError();
         }
         i->radius = i->radius + probeRadius;
      }
   }

   std::cout << "The molecule has " << atoms.size() << " atoms.\n";

   // The data structure for the molecule.
   Molecule molecule;
   for (std::size_t n = 0; n != atoms.size(); ++n) {
      molecule.insert(n, atoms[n]);
   }

   // Read the points.
   std::vector<Point> points;
   {
      std::ifstream in(parser.getArgument().c_str());
      in >> points;
   }
   std::cout << "There are " << points.size() << " points.\n";

   std::cout << "Computing the signed distance...\n" << std::flush;
   ads::Timer timer;
   timer.tic();

   // Compute the signed distance.
   std::vector<NumberType> signedDistance(points.size());
   for (std::size_t n = 0; n != points.size(); ++n) {
      signedDistance[n] = molecule.computeSignedDistance(points[n]);
   }

   double elapsedTime = timer.toc();
   std::cout << "Done.  Time = " << elapsedTime << "\n" << std::flush;


   // Write the signed distance.
   std::cout << "Writing the signed distance...\n" << std::flush;
   {
      std::ofstream out(parser.getArgument().c_str());
      out << "Distance\n" << signedDistance;
   }
   std::cout << "Done.\n" << std::flush;

   // There should be no more arguments.
   assert(parser.areArgumentsEmpty());

   return 0;
}
