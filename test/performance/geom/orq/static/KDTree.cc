// -*- C++ -*-

#include "stlib/geom/orq/KDTree.h"
#include "stlib/ads/functor/Dereference.h"
#include "stlib/ads/timer.h"
#include "stlib/ads/utility/ParseOptionsArguments.h"
#include "stlib/ext/vector.h"

#include <fstream>
#include <iterator>

using namespace stlib;

std::string programName;

void
exitOnError()
{
  std::cerr
      << "Bad arguments.  Usage:\n"
      << programName << " points.txt searchRadius\n"
      << "The first argument is a file that contains a list of points.\n"
      << "The second is the search radius for the ORQ's.\n"
      << "\nExiting...\n";
  exit(1);
}

int
main(int argc, char* argv[])
{
  USING_STLIB_EXT_VECTOR_IO_OPERATORS;

  const std::size_t Dimension = 3;
  typedef float Float;
  typedef std::array<Float, Dimension> Point;
  typedef geom::BBox<Float, Dimension> BBox;
  typedef std::vector<Point>::const_iterator Record;

  // Parse the program name and options.
  ads::ParseOptionsArguments parser(argc, argv);
  programName = parser.getProgramName();

  if (parser.getNumberOfArguments() != 2) {
    std::cerr << "Bad number of required arguments.\n"
              << "You gave the arguments:\n";
    parser.printArguments(std::cerr);
    exitOnError();
  }

  // Read the vector of points.
  std::vector<Point> points;
  {
    std::ifstream input(parser.getArgument().c_str());
    input >> points;
  }

  // Print information on the bounding box for the points.
  std::cout << "The bounding box for the points:\n"
            << geom::specificBBox<BBox>(points.begin(), points.end()) << "\n";

  Float searchRadius;
  parser.getArgument(&searchRadius);

  typedef geom::KDTree<Dimension, ads::Dereference<Record> > Orq;

  ads::Timer timer;
  timer.tic();
  Orq orq(points.begin(), points.end());
  const double initializationTime = timer.toc();

  std::size_t numNeighbors = 0;
  timer.tic();
  {
    std::vector<Record> neighbors;
    std::back_insert_iterator<std::vector<Record> > output(neighbors);
    BBox window;
    for (std::size_t i = 0; i != points.size(); ++i) {
      window.lower = window.upper = points[i];
      offset(&window, searchRadius);
      orq.computeWindowQuery(output, window);
      numNeighbors += neighbors.size();
      neighbors.clear();
    }
  }
  const double queryTime = timer.toc();

  std::cout << "Number of neighbors = " << numNeighbors << '\n'
            << "Average number of neighbors per query = "
            << double(numNeighbors) / points.size() << '\n'
            << "Initialization time = " << initializationTime << " seconds.\n"
            << "Query time = " << queryTime << " seconds.\n"
            << "Time per neighbor = " << queryTime / numNeighbors * 1e9
            << " nanoseconds\n";

  return 0;
}



