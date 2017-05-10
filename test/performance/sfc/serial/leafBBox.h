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
  typedef geom::BBox<Traits::Float, Traits::Dimension> Cell;
  typedef sfc::AdaptiveCells<Traits, Cell, false> Cells;
  typedef Cells::Point Point;
  typedef Cells::Float Float;
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

  std::vector<Point> points(numPoints);
  // The points are uniformly-distributed random points.
  for (std::size_t i = 0; i != points.size(); ++i) {
    for (std::size_t j = 0; j != Dimension; ++j) {
      points[i][j] = random();
    }
  }

  // Build the cells.
  geom::BBox<Float, Dimension> const domain =
    geom::specificBBox<geom::BBox<Float, Dimension> >
    (points.begin(), points.end());
  Cells cells(domain, minCellLength);
  std::cout << "numPoints = " << numPoints << '\n'
            << "minCellLength = " << minCellLength << '\n'
            << "numLevels = " << cells.numLevels() << '\n';

  timer.tic();
  cells.buildCells(&points);
  elapsedTime = timer.toc();
  std::cout << "time build() = " << elapsedTime << '\n';
  std::cout << "size() = " << cells.size() << '\n';

  timer.tic();
  result += cells.coarsen(sfc::LevelGreaterThan{cells.numLevels() - 1});
  elapsedTime = timer.toc();
  std::cout << "time coarsen() = " << elapsedTime << '\n';
  std::cout << "size() = " << cells.size() << '\n';

  std::cout << "Meaningless result = " << result << '\n';

  return 0;
}
