// -*- C++ -*-

#include "stlib/particle/codes.h"

#include "stlib/ads/timer/Timer.h"

#include <iostream>

#include <cassert>

using namespace stlib;

//! The main loop.
int
main()
{
  typedef particle::Morton<Float, Dimension, false> Morton;
  typedef Morton::Point Point;

  ads::Timer timer;
  double elapsedTime;
  std::size_t volatile result = 0;

  const geom::BBox<Float, Dimension> Domain = {ext::filled_array<Point>(0),
                                               ext::filled_array<Point>(1)
                                              };

  Float cellLength = 1;
  const std::size_t MaxLevel =
    std::min((std::numeric_limits<std::size_t>::digits - 1) / Dimension,
             std::size_t(std::numeric_limits<Float>::digits));
  std::cout << "Levels";
  for (std::size_t numLevels = 0; numLevels <= MaxLevel; ++numLevels) {
    std::cout << ',' << numLevels;
  }
  std::cout << "\nCodeToCoordinates";
  for (std::size_t numLevels = 0; numLevels <= MaxLevel;
       ++numLevels, cellLength *= 0.5) {
    Morton morton(Domain, cellLength);

    std::size_t count = 1000;
    do {
      count *= 2;
      timer.tic();
      for (std::size_t i = 0; i != count; ++i) {
        result += stlib::ext::sum(morton.coordinates(i));
      }
      elapsedTime = timer.toc();
    }
    while (elapsedTime < 0.1);

    std::cout << ',' << elapsedTime / count * 1e9;
  }
  std::cout << '\n';

  std::cout << "\nMeaningless result = " << result << "\n";

  return 0;
}
