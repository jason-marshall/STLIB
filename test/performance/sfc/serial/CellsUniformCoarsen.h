// -*- C++ -*-

#include "stlib/sfc/UniformCells.h"

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
      << " [-h]\n";
  exit(0);
}

int
main(int argc, char* argv[])
{
  typedef sfc::Traits<Dimension> Traits;
  typedef geom::BBox<Traits::Float, Traits::Dimension> Cell;
  typedef sfc::UniformCells<Traits, Cell, true> Cells;
  typedef Cells::Point Point;
  typedef numerical::ContinuousUniformGeneratorOpen<>
  ContinuousUniformGenerator;

  // Parse the options.
  ads::ParseOptionsArguments parser(argc, argv);
  programName = parser.getProgramName();
  if (parser.getOption('h')) {
    helpMessage();
  }

  ContinuousUniformGenerator::DiscreteUniformGenerator generator;
  ContinuousUniformGenerator random(&generator);

  ads::Timer timer;
  double elapsedTime;
  std::size_t result = 0;

  std::cout <<
            "num levels, num objects, num cells, time per cell (nanoseconds)\n";
  for (std::size_t numLevels = 4; numLevels <= 8; ++numLevels) {
    Cells cells(ext::filled_array<Point>(0), ext::filled_array<Point>(1),
                numLevels);
    std::vector<Point> objects(std::size_t(1) << (Dimension * numLevels));
    // The objects are uniformly-distributed random points.
    for (std::size_t i = 0; i != objects.size(); ++i) {
      for (std::size_t j = 0; j != Dimension; ++j) {
        objects[i][j] = random();
      }
    }
    cells.buildCells(&objects);
    const std::size_t numCells = cells.size();
    std::cout << numLevels
              << ", " << objects.size()
              << ", " << numCells;
    timer.tic();
    cells.coarsen();
    elapsedTime = timer.toc();
    std::cout  << ", " << elapsedTime / numCells * 1e9 << '\n';
    for (std::size_t i = 0; i != cells.size(); ++i) {
      result += cells.delimiter(i + 1) - cells.delimiter(i);
    }
  }
  std::cout << "Meaningless result = " << result << '\n';

  return 0;
}
