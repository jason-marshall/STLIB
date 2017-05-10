// -*- C++ -*-

#include "stlib/levelSet/MolecularSurfaces.h"
#include "stlib/ads/utility/ParseOptionsArguments.h"
#include "stlib/ads/timer.h"
#include "stlib/ext/vector.h"
#include "stlib/geom/mesh/iss/file_io.h"
#include "stlib/geom/mesh/iss/distinct_points.h"

#include <iostream>
#include <fstream>

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
    << "     input.xyzr [boundary.vtu]\n"
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

  //
  // Parse the program options and arguments.
  //
  ads::ParseOptionsArguments parser(argc, argv);

  programName = parser.getProgramName();

  // Verbose
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
  // Calculate the domain.
  //
  if (verbose) {
    std::cout << "Calculating the level set..." << std::endl;
  }
  timer.tic();
  std::vector<std::array<Number, Dimension> > vertices;
  levelSet::MolecularSurfaces<Number, Dimension, PatchExtent> 
    ms(balls, probeRadius, targetGridSpacing);
  std::pair<Number, Number> cb;
  if (parser.getOption("m")) {
    cb = ms.vanDerWaals(std::back_inserter(vertices));
  }
  else if (parser.getOption("a")) {
    cb = ms.solventAccessible(std::back_inserter(vertices));
  }
  else if (parser.getOption("e")) {
    cb = ms.solventExcluded(std::back_inserter(vertices));
  }
  else if (parser.getOption("ec")) {
    cb = ms.solventExcludedCavities(std::back_inserter(vertices));
  }
  else if (parser.getOption("ac")) {
    std::cerr << "Error: Not yet implemented.\n";
    exitOnError();
  }
  else {
    std::cerr << "Error: You must choose the surface to generate.\n";
    exitOnError();
  }
  if (verbose) {
    std::cout << "  Time = " << timer.toc() << std::endl;
    std::cout << "  content = " << cb.first << ", boundary = " << cb.second
              << "." << std::endl;
  }

  // Check that we parsed all of the options.
  if (! parser.areOptionsEmpty()) {
    std::cerr << "Error.  Unmatched options:\n";
    parser.printOptions(std::cerr);
    exitOnError();
  }

  //
  // Compute the content.
  //
  if (! parser.areArgumentsEmpty()) {
    if (verbose) {
      std::cout << "Building the boundary mesh..." << std::endl;
    }
    timer.tic();
    assert(vertices.size() % Dimension == 0);
    std::vector<std::array<std::size_t, Dimension> >
      indexedSimplices(vertices.size() / Dimension);
    std::size_t n = 0;
    for (std::size_t i = 0; i != indexedSimplices.size(); ++i) {
      for (std::size_t j = 0; j != Dimension; ++j) {
        indexedSimplices[i][j] = n++;
      }
    }
    // Build a mesh from the vertices and simplices.
    geom::IndSimpSet<Dimension, Dimension-1, Number>
      mesh(vertices, indexedSimplices);
    if (verbose) {
      std::cout << "  Time = " << timer.toc() << std::endl;
    }

    // Remove the duplicate vertices.
    if (verbose) {
      std::cout << "Removing duplicate vertices..." << std::endl;
    }
    timer.tic();
    removeDuplicateVertices(&mesh);
    if (verbose) {
      std::cout << "  Time = " << timer.toc() << std::endl;
    }

    // Write the boundary file.
    if (verbose) {
      std::cout << "Writing the boundary mesh..." << std::endl;
    }
    timer.tic();
    std::ofstream file(parser.getArgument().c_str());
    if (! file.good()) {
      std::cerr << "Could not open the boundary output file.";
      exitOnError();
    }
    geom::writeVtkXml(file, mesh);
    if (verbose) {
      std::cout << "  Time = " << timer.toc() << std::endl;
    }
  }

  if (verbose) {
    std::cout << "Volume and surface area:\n";
  }
  std::cout << cb.first << ' ' << cb.second << '\n';

  return 0;
}
