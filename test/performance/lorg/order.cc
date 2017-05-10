// -*- C++ -*-

#include "stlib/lorg/order.h"

#include "stlib/ads/timer/Timer.h"
#include "stlib/ext/vector.h"
#include "stlib/numerical/random/uniform/ContinuousUniformGenerator.h"

#include <iostream>

using namespace stlib;

int
main()
{
  typedef float Float;
  const std::size_t Dimension = 3;

  typedef std::array<Float, Dimension> Point;
  typedef numerical::ContinuousUniformGeneratorOpen<>
  ContinuousUniformGenerator;

  ContinuousUniformGenerator::DiscreteUniformGenerator generator;
  ContinuousUniformGenerator random(&generator);

  ads::Timer timer;
  double elapsedTime;
  std::size_t result = 0;

  // The columns indicate the number of bits in the code.
  std::cout << ','
            << std::numeric_limits<unsigned char>::digits << ','
            << std::numeric_limits<unsigned short>::digits << ','
            << std::numeric_limits<unsigned>::digits << ','
            << std::numeric_limits<std::size_t>::digits << '\n';

  std::vector<std::size_t> indices;
  for (std::size_t size = 1000; size <= 10000000; size *= 10) {
    // The row labels indicate the number of points.
    std::cout << size;
    // The positions are uniformly-distributed random points.
    std::vector<Point> positions(size);
    for (std::size_t i = 0; i != positions.size(); ++i) {
      for (std::size_t j = 0; j != Dimension; ++j) {
        positions[i][j] = random();
      }
    }

    // Warm up.
    lorg::mortonOrder<unsigned char>(positions, &indices);

    // Record the time in nanoseconds per element.
    timer.tic();
    lorg::mortonOrder<unsigned char>(positions, &indices);
    elapsedTime = timer.toc();
    result += stlib::ext::sum(indices);
    std::cout << ',' << 1e9 * elapsedTime / size;

    timer.tic();
    lorg::mortonOrder<unsigned short>(positions, &indices);
    elapsedTime = timer.toc();
    result += stlib::ext::sum(indices);
    std::cout << ',' << 1e9 * elapsedTime / size;

    timer.tic();
    lorg::mortonOrder<unsigned>(positions, &indices);
    elapsedTime = timer.toc();
    result += stlib::ext::sum(indices);
    std::cout << ',' << 1e9 * elapsedTime / size;

    timer.tic();
    lorg::mortonOrder<std::size_t>(positions, &indices);
    elapsedTime = timer.toc();
    result += stlib::ext::sum(indices);
    std::cout << ',' << 1e9 * elapsedTime / size;

    std::cout << '\n';
  }

  std::cout << "Meaningless result = " << result << "\n";

  return 0;
}
