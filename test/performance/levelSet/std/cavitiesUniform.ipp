// -*- C++ -*-

#include "stlib/levelSet/count.h"
#include "stlib/levelSet/marchingSimplices.h"
#include "stlib/levelSet/positiveDistance.h"
#include "stlib/levelSet/solventAccessibleCavities.h"
#include "stlib/levelSet/solventExcludedCavities.h"
#include "stlib/levelSet/flood.h"
#include "stlib/ads/utility/ParseOptionsArguments.h"
#include "stlib/ads/timer.h"
#include "stlib/ext/vector.h"
#include "stlib/geom/mesh/iss/file_io.h"
#include "stlib/geom/mesh/iss/distinct_points.h"

#include <iostream>
#include <fstream>

USING_STLIB_EXT_ARRAY_IO_OPERATORS;
using namespace stlib;

// Global variables.

//! The program name.
std::string programName;

// Local functions.

//! Exit with an error message and usage information.
void
exitOnError() {
   std::cerr
      << "Bad arguments.  Usage:\n"
      << programName << " [-m] [-e] [-ec] [-ac] [-r=probeRadius] [-s=gridSpacing]\n"
      << "    input.xyzr [levelSet.vti]\n"
      << "Distance is measured in Angstroms.\n"
      << "The -m option computes the molecular surface.\n"
      << "The -e option computes the solvent-excluded surface.\n"
      << "The -ec option computes the solvent-excluded cavities.\n"
      << "The -ac option computes the solvent-accessible cavities.\n"
      << "The default probe radius is 1.4 Angstroms.\n"
      << "The default grid spacing is 0.1 Angstroms.\n"
      << "The input file is a sequence of x, y, z coordinates and radius.\n"
      << "The output file (which is written as a VTK XML image file) is optional.\n"
      << "\nExiting...\n";
   exit(1);
}

int
main(int argc, char* argv[]) {
   //
   // Constants and types.
   //
   typedef float Number;
   typedef geom::BBox<Number, Dimension> BBox;
   typedef levelSet::GridUniform<Number, Dimension> Grid;

   //
   // Parse the program options and arguments.
   //
   ads::ParseOptionsArguments parser(argc, argv);

   programName = parser.getProgramName();

   // Verbose.
   const bool verbose = parser.getOption("v");

   // Probe radius.
   Number probeRadius = 1.4;
   parser.getOption("r", &probeRadius);
   if (probeRadius < 0) {
      std::cerr << "The probe radius may not be negative.";
      exitOnError();
   }

   // Grid spacing.
   Number targetGridSpacing = 0.1;
   parser.getOption("s", &targetGridSpacing);
   if (targetGridSpacing <= 0) {
      std::cerr << "The target grid spacing must be positive.";
      exitOnError();
   }

   if (parser.getNumberOfArguments() < 1 ||
       parser.getNumberOfArguments() > 2) {
      std::cerr << "Bad number of required arguments.\n"
                << "You gave the arguments:\n";
      parser.printArguments(std::cerr);
      exitOnError();
   }

   // Read the set of balls.
   if (verbose) {
      std::cout << "Reading the xyzr file..." << std::endl;
   }
   ads::Timer timer;
   timer.tic();
   std::vector<geom::Ball<Number, Dimension> > balls;
   {
      std::ifstream file(parser.getArgument().c_str());
      if (! file.good()) {
         std::cerr << "Could not open the input file.";
         exitOnError();
      }
      stlib::ext::readElements(file, &balls);
      if (balls.empty()) {
         std::cerr << "The set of balls is empty.\n";
         exitOnError();
      }
   }
   if (verbose) {
      std::cout << "  Time = " << timer.toc() << std::endl;
   }

   //
   // Determine an appropriate grid and domain.
   //
   if (verbose) {
      std::cout << "Constructing the grid..." << std::endl;
   }
   timer.tic();
   // Place a bounding box around the balls.
   BBox domain = geom::specificBBox<BBox>(balls.begin(), balls.end());
   // Expand by the probe radius so that we can determine the global cavities.
   // add the target grid spacing to get one more grid point.
   offset(&domain, probeRadius + targetGridSpacing);
   // Make the grid.
   Grid grid(domain, targetGridSpacing);
   if (verbose) {
      std::cout << "  Time = " << timer.toc() << std::endl;
      std::cout << "  Lower corner = " << grid.lowerCorner << '\n'
                << "  Spacing = " << grid.spacing << '\n'
                << "  Extents = " << grid.extents() << '\n';
   }

   //
   // Calculate the cavities.
   //
   if (verbose) {
      std::cout << "Calculating the level set..." << std::endl;
   }
   timer.tic();
   if (parser.getOption("m")) {
      levelSet::positiveDistance(&grid, balls, Number(0),
                                     2 * targetGridSpacing);
      levelSet::floodFill(&grid, 2 * targetGridSpacing, 2 * targetGridSpacing);
   }
   else if (parser.getOption("e")) {
      levelSet::solventExcluded(&grid, balls, probeRadius);
      levelSet::floodFill(&grid, probeRadius, probeRadius);
   }
   else if (parser.getOption("ec")) {
      // When subtracting the atoms, compute the distance up to 
      // two grid points past the surface each atom.
      const Number maxDistance = 2 * grid.spacing;
      levelSet::solventExcludedCavities(&grid, balls, probeRadius, maxDistance);
      levelSet::floodFill(&grid, probeRadius, probeRadius);
   }
   else if (parser.getOption("ac")) {
      levelSet::solventAccessibleCavities(&grid, balls, probeRadius);
      levelSet::floodFill(&grid, probeRadius, probeRadius);
   }
   else {
      std::cerr << "Error: You must choose the surface to generate.\n";
      exitOnError();
   }
   if (verbose) {
      std::cout << "  Time = " << timer.toc() << std::endl;
   }

   // Check that we parsed all of the options.
   if (! parser.areOptionsEmpty()) {
      std::cerr << "Error.  Unmatched options:\n";
      parser.printOptions(std::cerr);
      exitOnError();
   }

   //
   // Write the level set output file if a file name was specified.
   //
   if (! parser.areArgumentsEmpty()) {
      if (verbose) {
         std::cout << "Writing the level set..." << std::endl;
      }
      timer.tic();
      std::ofstream file(parser.getArgument().c_str());
      if (! file.good()) {
         std::cerr << "Could not open the level set output file.";
         exitOnError();
      }
      writeVtkXml(grid, file);
      if (verbose) {
         std::cout << "  Time = " << timer.toc() << std::endl;
      }
   }   

   return 0;
}
