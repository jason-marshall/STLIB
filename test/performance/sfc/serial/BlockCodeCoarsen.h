// -*- C++ -*-

#include "stlib/sfc/BlockCode.h"

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
      << programName << " [-l=L] [-m=M] [-n=N] [-h]\n"
    "-l: The number of levels of refinement.\n"
    "-m: The maximum number of objects per cell.\n"
    "-n: The number of points.\n";
  exit(0);
}

int
main(int argc, char* argv[])
{
  typedef sfc::Traits<Dimension> Traits;
  typedef sfc::BlockCode<Traits> BlockCode;
  typedef BlockCode::Code Code;
  typedef BlockCode::Point Point;
  typedef numerical::ContinuousUniformGeneratorOpen<>
  ContinuousUniformGenerator;

  // Parse the options.
  ads::ParseOptionsArguments parser(argc, argv);
  programName = parser.getProgramName();
  if (parser.getOption('h')) {
    helpMessage();
  }
  std::size_t numLevels = BlockCode::MaxLevels;
  parser.getOption('l', &numLevels);
  if (numLevels > BlockCode::MaxLevels) {
    std::cout << "Error: The maximum number of levels of refinement is "
              << BlockCode::MaxLevels << '\n';
    helpMessage();
  }
  std::size_t maxObjectsPerCell = 128;
  parser.getOption('m', &maxObjectsPerCell);
  std::size_t numObjects = 1000000;
  parser.getOption('n', &numObjects);

  ContinuousUniformGenerator::DiscreteUniformGenerator generator;
  ContinuousUniformGenerator random(&generator);

  BlockCode blockCode(ext::filled_array<Point>(0), ext::filled_array<Point>(1),
                      numLevels);

  // The codes for uniformly-distributed random points.
  std::vector<Code> objectCodes(numObjects);
  {
    Point p;
    for (std::size_t i = 0; i != objectCodes.size(); ++i) {
      for (std::size_t j = 0; j != p.size(); ++j) {
        p[j] = random();
      }
      objectCodes[i] = blockCode.code(p);
    }
  }
  std::sort(objectCodes.begin(), objectCodes.end());

  std::vector<Code> cellCodes;
  ads::Timer timer;
  timer.tic();
  coarsen(blockCode, objectCodes, &cellCodes, maxObjectsPerCell);
  double const elapsedTime = timer.toc();

  std::cout << "number of levels of refinement = " << numLevels << '\n'
            << "maximum objects per cell = " << maxObjectsPerCell << '\n'
            << "number of objects = " << objectCodes.size() << '\n'
            << "number of cells = " << cellCodes.size() << '\n'
            << "elapsed time = " << elapsedTime << '\n'
            << "time per object = " << elapsedTime / objectCodes.size() * 1e9
            << " nanoseconds\n";

  return 0;
}
