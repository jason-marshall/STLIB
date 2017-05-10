// -*- C++ -*-

#include "stlib/sfc/AdaptiveCells.h"

#include "stlib/ads/functor/Identity.h"
#include "stlib/ads/timer/Timer.h"
#include "stlib/ads/utility/ParseOptionsArguments.h"
#include "stlib/numerical/random/uniform/ContinuousUniformGenerator.h"

#include <iostream>

using namespace stlib;

// The program name.
std::string programName;

// Exit with a usage message.
void
helpMessage()
{
  std::cout
      << "Usage:\n"
      << programName
      << " [-h] [-p=P] [-l=L]\n"
      << "-p: The number of points. The default is 1000.\n"
      << "-l: The minimum cell length. The default is 0.01.\n";
  exit(0);
}

int
main(int argc, char* argv[])
{
  typedef sfc::Traits<Dimension> Traits;
  typedef sfc::AdaptiveCells<Traits, void, true> Tree;
  typedef Tree::Point Point;
  typedef Tree::Float Float;
  typedef numerical::ContinuousUniformGeneratorOpen<>
  ContinuousUniformGenerator;

  // Parse the options.
  ads::ParseOptionsArguments parser(argc, argv);
  programName = parser.getProgramName();
  if (parser.getOption('h')) {
    helpMessage();
  }
  std::size_t numPoints = 1000;
  parser.getOption('p', &numPoints);
  Float minCellLength = 0.01;
  parser.getOption('l', &minCellLength);

  ContinuousUniformGenerator::DiscreteUniformGenerator generator;
  ContinuousUniformGenerator random(&generator);

  ads::Timer timer;
  double elapsedTime;
  std::size_t result = 0;

  // Uniformly-distributed random points.
  std::vector<Point> points(numPoints);
  for (std::size_t i = 0; i != points.size(); ++i) {
    for (std::size_t j = 0; j != Dimension; ++j) {
      points[i][j] = random();
    }
  }

  // Make the orthant trie.
  geom::BBox<Float, Dimension> const domain =
    geom::specificBBox<geom::BBox<Float, Dimension> >
    (points.begin(), points.end());
  Tree tree(domain, minCellLength);
  std::cout << "numPoints = " << numPoints << '\n'
            << "minCellLength = " << minCellLength << '\n'
            << "numLevels = " << tree.numLevels() << '\n';
  timer.tic();
  tree.buildCells(&points);
  elapsedTime = timer.toc();
  std::cout << "time build() = " << elapsedTime << '\n';

  // Iterate over the cells and count the objects.
  {
    std::size_t count = 0;
    for (std::size_t i = 0; i != tree.size(); ++i) {
      count += tree.delimiter(i + 1) - tree.delimiter(i);
    }
    assert(count == points.size());
  }

  std::cout << "size() = " << tree.size() << '\n'
            << "Meaningless result = " << result << '\n';

  return 0;
}
