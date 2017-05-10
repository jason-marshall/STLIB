// -*- C++ -*-

#include "stlib/levelSet/count.h"
#include "stlib/levelSet/marchingSimplices.h"
#include "stlib/levelSet/powerDistance.h"
#include "stlib/levelSet/solventExcludedCavities.h"
#include "stlib/levelSet/solventAccessibleCavities.h"
#include "stlib/levelSet/flood.h"
#include "stlib/ads/utility/ParseOptionsArguments.h"
#include "stlib/ads/timer.h"
#include "stlib/ext/vector.h"
#include "stlib/geom/mesh/iss/file_io.h"
#include "stlib/geom/mesh/iss/distinct_points.h"

#include <iostream>
#include <fstream>

USING_STLIB_EXT_VECTOR_IO_OPERATORS;
using namespace stlib;

//
// Types.
//
typedef float Number;
typedef std::array<Number, Dimension> Point;
typedef geom::BBox<Number, Dimension> BBox;
typedef levelSet::Grid<Number, Dimension, PatchExtent> Grid;
typedef Grid::IndexList IndexList;

// Global variables.

//! The program name.
std::string programName;
bool verbose;

// Local functions.

// Exit with an error message and usage information.
void
exitOnError() {
   std::cerr
      << "Bad arguments.  Usage:\n"
      << programName << " [-m] [-e] [-es] [-ec] [-ac] [-r=probeRadius] [-s=gridSpacing]\n"
      << "     input.xyzr [boundary.vtu [levelSet.vti]]\n"
      << "Distance is measured in Angstroms.\n"
      << "The -m option computes the molecular surface.\n"
      << "The -e option computes the solvent-excluded surface.\n"
      << "The -es option computes the solvent-excluded surface using seeds.\n"
      << "The -ec option computes the solvent-excluded cavities.\n"
      << "The -ecs option computes the solvent-excluded cavities using seeds.\n"
      << "The -ac option computes the solvent-accessible cavities.\n"
      << "The default probe radius is 1.4 Angstroms.\n"
      << "The default grid spacing is 0.1 Angstroms.\n"
      << "The input file is a sequence of x, y, z coordinates and radius.\n"
      << "The output file (which is written as a VTK XML image file) is optional.\n"
      << "\nExiting...\n";
   exit(1);
}


void
buildBoundary
(ads::ParseOptionsArguments* parser,
 const std::vector<std::array<Number, Dimension> >& vertices) {
   // Check if they specified an output file for the boundary.
   if (parser->areArgumentsEmpty()) {
      return;
   }

   if (verbose) {
      std::cout << "Building the boundary mesh..." << std::endl;
   }
   ads::Timer timer;
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
   std::ofstream file(parser->getArgument().c_str());
   if (! file.good()) {
      std::cerr << "Could not open the boundary output file.";
      exitOnError();
   }
   // use geom::mesh::simplicial version (writes to UnstructuredGrid VTK file)
   geom::writeVtkXml(file, mesh);
   if (verbose) {
      std::cout << "  Time = " << timer.toc() << std::endl;
   }
}


std::pair<Number, Number>
contentAndBoundary(ads::ParseOptionsArguments* parser, const Grid& grid) {
   Number content, boundary;
   std::vector<std::array<Number, Dimension> > vertices;
   if (verbose) {
      std::cout << "Computing the content..." << std::endl;
   }
   ads::Timer timer;
   timer.tic();
   if (parser->areArgumentsEmpty()) {
      levelSet::contentAndBoundary(grid, &content, &boundary);
   }
   else {
      levelSet::contentAndBoundary(grid, &content, &boundary,
                                   std::back_inserter(vertices));
   }
   if (verbose) {
      std::cout << "  Time = " << timer.toc() << std::endl;
      std::cout << "  content = " << content << ", boundary = " << boundary
                << "." << std::endl;
   }

   buildBoundary(parser, vertices);

   return std::make_pair(content, boundary);
}


void
contentAndBoundary(ads::ParseOptionsArguments* parser, const Grid& grid,
                   std::vector<Number>* content,
                   std::vector<Number>* boundary) {
   std::vector<std::array<Number, Dimension> > vertices;
   if (verbose) {
      std::cout << "Computing the content..." << std::endl;
   }
   ads::Timer timer;
   timer.tic();
   if (parser->areArgumentsEmpty()) {
      levelSet::contentAndBoundary(grid, content, boundary);
   }
   else {
      levelSet::contentAndBoundary(grid, content, boundary,
                                   std::back_inserter(vertices));
   }
   if (verbose) {
      std::cout << "  Time = " << timer.toc() << std::endl;
      std::cout << "  content = \n" << *content << "boundary = \n" << *boundary
                << "." << std::endl;
   }

   buildBoundary(parser, vertices);
}


void
writeLevelSet(ads::ParseOptionsArguments* parser, const Grid& grid) {
   if (! parser->areArgumentsEmpty()) {
      if (verbose) {
         std::cout << "Writing the level set..." << std::endl;
      }
      ads::Timer timer;
      timer.tic();
      std::ofstream file(parser->getArgument().c_str());
      if (! file.good()) {
         std::cerr << "Could not open the level set output file.";
         exitOnError();
      }
      // use levelSet::Grid version (writes to ImageData VTK file)
      writeVtkXml(grid, file);
      if (verbose) {
         std::cout << "  Time = " << timer.toc() << std::endl;
      }
   }   
}


int
main(int argc, char* argv[]) {
   //
   // Parse the program options and arguments.
   //
   ads::ParseOptionsArguments parser(argc, argv);

   programName = parser.getProgramName();

   // Verbose.
   verbose = parser.getOption("v");

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
       parser.getNumberOfArguments() > 3) {
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
   // Calculate the domain.
   //
   bool useComponents = false;
   if (verbose) {
      std::cout << "Calculating the level set..." << std::endl;
   }
   timer.tic();
   if (parser.getOption("m")) {
      ads::Timer t;
      t.tic();
      levelSet::negativePowerDistance(&grid, balls);
      if (verbose) {
         std::cout << "  Time for negative distance only = " << t.toc()
                   << std::endl;
      }

      Number maxRadius = 0;
      for (std::size_t i = 0; i != balls.size(); ++i) {
         if (balls[i].radius > maxRadius) {
            maxRadius = balls[i].radius;
         }
      }
      const Number lower = - maxRadius * maxRadius;
      // The distance will be correctly computed up to delta.
      const Number delta = 1.1 * grid.spacing * std::sqrt(Number(Dimension));
      // (mR + d) * (mR + d) - mR * mR;
      const Number upper = 2 * maxRadius * delta + delta * delta;
      levelSet::floodFillInterval(&grid, lower, upper, lower, upper);
   }
   else if (parser.getOption("a")) {
      std::vector<geom::Ball<Number, Dimension> > offset(balls);
      for (std::size_t i = 0; i != offset.size(); ++i) {
         offset[i].radius += probeRadius;
      }
      levelSet::negativePowerDistance(&grid, offset);
      Number maxRadius = 0;
      for (std::size_t i = 0; i != offset.size(); ++i) {
         if (offset[i].radius > maxRadius) {
            maxRadius = offset[i].radius;
         }
      }
      const Number lower = - maxRadius * maxRadius;
      // The distance will be correctly computed up to delta.
      const Number delta = 1.1 * grid.spacing * std::sqrt(Number(Dimension));
      // (mR + d) * (mR + d) - mR * mR;
      const Number upper = 2 * maxRadius * delta + delta * delta;
      levelSet::floodFillInterval(&grid, lower, upper, lower, upper);
   }
   else if (parser.getOption("e")) {
      levelSet::solventExcluded(&grid, balls, probeRadius);
      levelSet::floodFill(&grid, probeRadius, probeRadius);
   }
   else if (parser.getOption("es")) {
      levelSet::solventExcludedUsingSeeds(&grid, balls, probeRadius);
      levelSet::floodFill(&grid, probeRadius, probeRadius);
   }
   else if (parser.getOption("ec")) {
      // When subtracting the atoms, compute the distance up to 
      // two grid points past the surface each atom.
      const Number maxDistance = 2 * grid.spacing;
      levelSet::solventExcludedCavities(&grid, balls, probeRadius, maxDistance);
      levelSet::floodFill(&grid, probeRadius, probeRadius);
   }
   else if (parser.getOption("ecs")) {
      levelSet::solventExcludedCavitiesUsingSeeds(&grid, balls, probeRadius);
      levelSet::floodFill(&grid, probeRadius, probeRadius);
   }
   else if (parser.getOption("ac")) {
      useComponents = true;
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

   if (useComponents) {
      // Compute the content.
      std::vector<Number> content, boundary;
      contentAndBoundary(&parser, grid, &content, &boundary);
      // Write the level set output file if a file name was specified.
      writeLevelSet(&parser, grid);

      if (verbose) {
         std::cout << "Total volume and surface area:\n";
      }
      std::cout << stlib::ext::sum(content) << ' '
                << stlib::ext::sum(boundary) << '\n';
      if (verbose) {
         std::cout << "Component volume and surface area:\n";
      }
      std::cout << content.size() << '\n';
      for (std::size_t i = 0; i != content.size(); ++i) {
         std::cout << content[i] << ' ' << boundary[i] << '\n';
      }
   }
   else {
      // Compute the content.
      const std::pair<Number, Number> cb = contentAndBoundary(&parser, grid);
      // Write the level set output file if a file name was specified.
      writeLevelSet(&parser, grid);
      if (verbose) {
         std::cout << "Volume and surface area:\n";
      }
      std::cout << cb.first << ' ' << cb.second << '\n';
   }

   return 0;
}
