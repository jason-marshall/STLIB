// -*- C++ -*-

#include "stlib/ads/utility/ParseOptionsArguments.h"
#include "stlib/numerical/random/uniform/ContinuousUniformGenerator.h"
#include "stlib/performance/SimpleTimer.h"

#include <iostream>
#include <vector>

#include <cstdlib>

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
  typedef std::array<Code, Dimension> DiscretePoint;
  std::size_t const NumLevels =
    std::numeric_limits<Code>::digits / Dimension;
  std::cout << "Number of levels of refinement = " << NumLevels << ".\n";
  Code const Mask = (Code(1) << NumLevels) - 1;
  
  // Parse the options.
  ads::ParseOptionsArguments parser(argc, argv);
  programName = parser.getProgramName();
  if (parser.getOption('h')) {
    helpMessage();
  }

  numerical::DISCRETE_UNIFORM_GENERATOR_DEFAULT generator;
  std::vector<DiscretePoint> coords(1 << 20);
  for (std::size_t i = 0; i != coords.size(); ++i) {
    for (std::size_t j = 0; j != Dimension; ++j) {
      coords[i][j] = generator() & Mask;
    }
  }
  Order order;
  
  Code result = 0;
  stlib::performance::SimpleTimer timer;
  timer.start();
  for (std::size_t i = 0; i != coords.size(); ++i) {
    result += order.code(coords[i], NumLevels);
  }
  timer.stop();
  std::cout << "Time to generate code = "
            << timer.nanoseconds() / coords.size() << " ns.\n"
            << "Meaningless result = " << result << '\n';

  return 0;
}
