// -*- C++ -*-

// Dimension and Float must be defined before including this file.

#include "stlib/geom/kernel/content.h"
#include "stlib/performance/SimpleTimer.h"

#include <random>

int
main()
{
  using SimpleTimer = stlib::performance::SimpleTimer;

  using Point = std::array<Float, Dimension>;
  using Simplex = std::array<Point, Dimension + 1>;

  std::mt19937_64 generator;
  std::uniform_real_distribution<Float> distribution(0, 1);

  // Simplices with random vertices.
  std::vector<Simplex> simplices(1 << 16);
  std::cout << "Number of simplices = " << simplices.size() << '\n';
  for (Simplex& s: simplices) {
    for (Point& p: s) {
      for (Float& x: p) {
        x = distribution(generator);
      }
    }
  }

  Float content = 0;
  SimpleTimer timer;
  timer.start();
  for (Simplex const& s: simplices) {
    content += stlib::geom::computeContent(s);
  }
  timer.stop();
  std::cout << "Total content = " << content << '\n'
            << "Time to compute " << simplices.size()
            << " contents = " << timer.elapsed() << " seconds.\n"
            << "Time per content calculation = "
            << timer.nanoseconds() / simplices.size()
            << " nanoseconds.\n";

  return 0;
}



