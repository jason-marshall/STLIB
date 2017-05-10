// -*- C++ -*-

#include "stlib/particle/order.h"

#include "stlib/ads/timer/Timer.h"
#include "stlib/numerical/random/uniform/ContinuousUniformGenerator.h"

#include <iostream>

#include <cassert>

using namespace stlib;

// The main loop.
int
main()
{
  typedef particle::MortonOrder<Traits> MortonOrder;
  typedef numerical::ContinuousUniformGeneratorOpen<>
  ContinuousUniformGenerator;

  ContinuousUniformGenerator::DiscreteUniformGenerator generator;
  ContinuousUniformGenerator random(&generator);

  ads::Timer timer;
  double elapsedTime;
  std::size_t result = 0;

  const geom::BBox<Float, Dimension> Domain = {ext::filled_array<Point>(0),
                                               ext::filled_array<Point>(1)
                                              };

  std::cout.setf(std::ios_base::fixed, std::ios_base::floatfield);
  std::cout << "<table border=\"1\">\n"
            << "<tr>"
            << "<th>Particles</th> <th>Radius</th> <th>Neighbors</th> <th>Time (ns)</th>\n";
  for (std::size_t numParticles = 10000; numParticles <= 1000000;
       numParticles *= 10) {
    // The particles are uniformly-distributed random points.
    std::vector<Point> particles(numParticles);
    for (std::size_t i = 0; i != particles.size(); ++i) {
      for (std::size_t j = 0; j != Dimension; ++j) {
        particles[i][j] = random();
      }
    }

    Float radius = std::pow(1. / particles.size(), 1. / Dimension);
    for (std::size_t i = 0; i != 3; ++i, radius *= 2) {
      MortonOrder morton(Domain, radius, 0);
#ifdef VERLET
      particle::VerletLists<MortonOrder> neighbors(morton);
#elif defined(ADJACENT)
      particle::AdjacentMask<MortonOrder> neighbors(morton);
#endif
      morton.setParticles(particles.begin(), particles.end());

      std::cout << "<tr><td>" << particles.size() << "</td>";
      std::cout.precision(std::max(2., 1. + std::ceil(-log10(radius))));
      std::cout << "<td>" << radius << "</td>";

      timer.tic();
      neighbors.findAllNeighbors();
      elapsedTime = timer.toc();

      const std::size_t numNeighbors = neighbors.numNeighbors();
      result += numNeighbors;

      std::cout.precision(1);
      std::cout << "<td>"
                << double(numNeighbors) / morton.particles.size()
                << "</td>";
      std::cout.precision(0);
      std::cout << "<td>" << elapsedTime * 1e9 / numNeighbors
                << "</td>\n";
    }
  }
  std::cout << "</table>\n";

  std::cout << "\nMeaningless result = " << result << "\n";

  return 0;
}
