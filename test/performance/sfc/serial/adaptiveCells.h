// -*- C++ -*-

#include "stlib/sfc/AdaptiveCells.h"

#include "stlib/ads/utility/ParseOptionsArguments.h"

#include <iostream>
#include <random>

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
  typedef stlib::sfc::Traits<Dimension> Traits;
  typedef stlib::sfc::AdaptiveCells<Traits, void, true> Cells;
  typedef Cells::Point Point;
  using Float = Traits::Float;

  // Parse the options.
  stlib::ads::ParseOptionsArguments parser(argc, argv);
  programName = parser.getProgramName();
  if (parser.getOption('h')) {
    helpMessage();
  }

  std::mt19937_64 engine;
  std::uniform_real_distribution<Float> uniform(0, 1);
  std::uniform_int_distribution<std::size_t> uniformInt(10000, 100000);

  std::array<double, 10> ratios;
  ratios.fill(0);
  constexpr std::size_t NumTests = 100;
  for (std::size_t i = 0; i != NumTests; ++i) {
    std::vector<Point> objects(uniformInt(engine));
    // The objects are uniformly-distributed random points.
    for (std::size_t i = 0; i != objects.size(); ++i) {
      for (std::size_t j = 0; j != Dimension; ++j) {
        objects[i][j] = uniform(engine);
      }
    }
  
    for (std::size_t shift = 0; shift != ratios.size(); ++shift) {
      std::size_t const maxObjectsPerCell = std::size_t(1) << shift;
      Cells const cells = stlib::sfc::adaptiveCells<Cells>(&objects,
                                                             maxObjectsPerCell);
      double const average = double(objects.size()) / cells.size();
      ratios[shift] += maxObjectsPerCell / average;
    }
  }

  std::cout << "max objects per cell, average ratio\n";
  for (std::size_t shift = 0; shift != ratios.size(); ++shift) {
    std::size_t const maxObjectsPerCell = std::size_t(1) << shift;
    ratios[shift] /= NumTests;
    std::cout << maxObjectsPerCell
              << ", " << ratios[shift] << '\n';
  }
  
  return 0;
}
