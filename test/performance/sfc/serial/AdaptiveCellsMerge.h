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
      << " [-h]\n";
  exit(0);
}

int
main(int argc, char* argv[])
{
  typedef sfc::Traits<Dimension> Traits;
  typedef geom::BBox<Traits::Float, Traits::Dimension> Cell;
  typedef sfc::AdaptiveCells<Traits, Cell, true> Cells;
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

    Cells cells1(ext::filled_array<Point>(0), ext::filled_array<Point>(1),
                 numLevels);
    std::vector<Point> objects1(std::size_t(1) << (Dimension * numLevels));
    // The objects are uniformly-distributed random points.
    for (std::size_t i = 0; i != objects1.size(); ++i) {
      for (std::size_t j = 0; j != Dimension; ++j) {
        objects1[i][j] = random();
      }
    }
    cells1.buildCells(&objects1);

    Cells cells2(ext::filled_array<Point>(0), ext::filled_array<Point>(1),
                 numLevels);
    std::vector<Point> objects2(std::size_t(1) << (Dimension * numLevels));
    // The objects are uniformly-distributed random points.
    for (std::size_t i = 0; i != objects2.size(); ++i) {
      for (std::size_t j = 0; j != Dimension; ++j) {
        objects2[i][j] = random();
      }
    }
    cells2.buildCells(&objects2);

    const std::size_t numCells = cells1.size() + cells2.size();
    std::cout << numLevels
              << ", " << objects1.size() + objects2.size()
              << ", " << numCells;
    timer.tic();
    cells1 += cells2;
    elapsedTime = timer.toc();
    std::cout  << ", " << elapsedTime / numCells * 1e9 << '\n';
    for (std::size_t i = 0; i != cells1.size(); ++i) {
      result += cells1.delimiter(i + 1) - cells1.delimiter(i);
    }
  }
  std::cout << "Meaningless result = " << result << '\n';

  return 0;
}
