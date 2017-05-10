// -*- C++ -*-

#ifndef __test_performance_levelSet_common_h__
#define __test_performance_levelSet_common_h__

#include "levelSet/flood.h"
#include "levelSet/marchingSimplices.h"

//
// Constants and types.
//
const std::size_t Dimension = 3;
typedef float Number;
typedef std::tr1::array<Number, Dimension> Point;
typedef geom::BBox<Number, Dimension> BBox;
typedef levelSet::Grid<Number, Dimension, levelSet::PatchExtent> Grid;
typedef Grid::IndexList IndexList;

std::string programName;
Number probeRadius = 1.4;
Number targetGridSpacing = 0.1;
bool verbose = false;

void
exitOnError()
{
  std::cerr
      << "Bad arguments.  Usage:\n"
      << programName << " [-v] [-s=gridSpacing]\n"
      << "     input.xyzr [boundary.vtu [levelSet.vti]]\n"
      << "Distance is measured in Angstroms.\n"
      << "In verbose mode (-v), timing and progress information will be printed.\n"
      << "The default grid spacing is 0.1 Angstroms.\n"
      << "The input file is a sequence of x, y, z coordinates and radius.\n"
      << "The output file (which is written as a VTK XML image file) is optional.\n"
      << "\nExiting...\n";
  exit(1);
}

void
parseNameAndOptions(ads::ParseOptionsArguments* parser)
{
  programName = parser->getProgramName();

  // Probe radius.
  parser->getOption("r", &probeRadius);
  if (probeRadius < 0) {
    std::cerr << "The probe radius may not be negative.";
    exitOnError();
  }

  // Verbose
  verbose = parser->getOption("v");

  // Grid spacing.
  parser->getOption("s", &targetGridSpacing);
  if (targetGridSpacing <= 0) {
    std::cerr << "The target grid spacing must be positive.";
    exitOnError();
  }

  // Check that we parsed all of the options.
  if (! parser->areOptionsEmpty()) {
    std::cerr << "Error.  Unmatched options:\n";
    parser->printOptions(std::cerr);
    exitOnError();
  }

  if (parser->getNumberOfArguments() < 1 ||
      parser->getNumberOfArguments() > 3) {
    std::cerr << "Bad number of required arguments.\n"
              << "You gave the arguments:\n";
    parser->printArguments(std::cerr);
    exitOnError();
  }
}

void
readBalls(ads::ParseOptionsArguments* parser,
          std::vector<geom::Ball<Number, Dimension> >* balls)
{
  if (verbose) {
    std::cout << "Reading the xyzr file..." << std::endl;
  }
  ads::Timer timer;
  timer.tic();
  {
    std::ifstream file(parser->getArgument().c_str());
    if (! file.good()) {
      std::cerr << "Could not open the input file.";
      exitOnError();
    }
    readElements(file, balls);
    if (balls->empty()) {
      std::cerr << "The set of balls is empty.\n";
      exitOnError();
    }
  }
  if (verbose) {
    std::cout << "  Time = " << timer.toc() << std::endl;
  }
}

void
floodFill(Grid* grid,
          const std::vector<geom::Ball<Number, Dimension> >& balls)
{
  if (verbose) {
    std::cout << "Flood fill..." << std::endl;
  }
  ads::Timer timer;
  timer.tic();
  Number maxRadius = 0;
  for (std::size_t i = 0; i != balls.size(); ++i) {
    if (balls[i].radius > maxRadius) {
      maxRadius = balls[i].radius;
    }
  }
  const Number lower = - maxRadius * maxRadius;
  // The distance will be correctly computed up to delta.
  const Number delta = 1.1 * grid->spacing * std::sqrt(Number(Dimension));
  // (mR + d) * (mR + d) - mR * mR;
  const Number upper = 2 * maxRadius * delta + delta * delta;
  levelSet::floodFillInterval(grid, lower, upper, lower, upper);
  if (verbose) {
    std::cout << "  Time = " << timer.toc() << std::endl;
  }
}

void
buildBoundary
(ads::ParseOptionsArguments* parser,
 const std::vector<std::tr1::array<Number, Dimension> >& vertices)
{
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
  std::vector<std::tr1::array<std::size_t, Dimension> >
  indexedSimplices(vertices.size() / Dimension);
  std::size_t n = 0;
  for (std::size_t i = 0; i != indexedSimplices.size(); ++i) {
    for (std::size_t j = 0; j != Dimension; ++j) {
      indexedSimplices[i][j] = n++;
    }
  }
  // Build a mesh from the vertices and simplices.
  geom::IndSimpSet < Dimension, Dimension - 1, Number >
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
  geom::writeVtkXml(file, mesh);
  if (verbose) {
    std::cout << "  Time = " << timer.toc() << std::endl;
  }
}


std::pair<Number, Number>
contentAndBoundary(ads::ParseOptionsArguments* parser, const Grid& grid)
{
  Number content, boundary;
  std::vector<std::tr1::array<Number, Dimension> > vertices;
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
                   std::vector<Number>* boundary)
{
  std::vector<std::tr1::array<Number, Dimension> > vertices;
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
writeLevelSet(ads::ParseOptionsArguments* parser, const Grid& grid)
{
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
    writeVtkXml(grid, file);
    if (verbose) {
      std::cout << "  Time = " << timer.toc() << std::endl;
    }
  }
}

#endif
