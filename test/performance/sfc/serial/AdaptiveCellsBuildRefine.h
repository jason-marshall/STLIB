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
  typedef sfc::BlockCode<Traits> BlockCode;
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

  std::vector<Point> objects(1000000);
  // The objects are uniformly-distributed random points.
  for (std::size_t i = 0; i != objects.size(); ++i) {
    for (std::size_t j = 0; j != Dimension; ++j) {
      objects[i][j] = random();
    }
  }

  std::cout << "1,000,000 objects\n";
  for (std::size_t levels = 8; levels <= BlockCode::MaxLevels; levels += 8) {
    std::cout << '\n' << levels << " levels of refinement.\n"
      "max objects per cell, time per object (nanoseconds)\n";
    for (std::size_t i = 0; i != 16; i += 2) {
      std::size_t const maxObjectsPerCell = std::size_t(1) << i;
      std::vector<Point> obj(objects);
      timer.tic();
      Cells cells(ext::filled_array<Point>(0), ext::filled_array<Point>(1),
                  levels);
      cells.buildCells(&obj, maxObjectsPerCell);
      elapsedTime = timer.toc();
      std::cout << maxObjectsPerCell
                << ", " << elapsedTime / objects.size() * 1e9 << '\n';
      for (std::size_t i = 0; i != cells.size(); ++i) {
        result += cells.delimiter(i + 1) - cells.delimiter(i);
      }
    }
  }
  std::cout << "Meaningless result = " << result << '\n';

  return 0;
}
