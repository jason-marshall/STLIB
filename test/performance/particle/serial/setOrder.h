// -*- C++ -*-

#include "stlib/particle/order.h"
#include "stlib/particle/traits.h"

#include "stlib/ads/functor/Identity.h"
#include "stlib/ads/timer/Timer.h"
#include "stlib/numerical/random/uniform/ContinuousUniformGenerator.h"

#include <iostream>

#include <cassert>

using namespace stlib;

template<typename _Traits>
class MortonOrder :
  public particle::MortonOrder<_Traits>
{
public:
  typedef typename _Traits::Float Float;
  typedef particle::MortonOrder<_Traits> Base;

  MortonOrder(const geom::BBox<Float, _Traits::Dimension>& domain,
              const Float interactionDistance, const Float padding) :
    Base(domain, interactionDistance, padding)
  {
  }

  using Base::morton;
  using Base::order;
};

// The main loop.
int
main()
{
  // A particle is just a point.
  typedef std::array<Float, Dimension> Point;
  typedef particle::PlainTraits<Point, ads::Identity<Point>, Dimension, Float>
  Traits;
  typedef MortonOrder<Traits> MortonOrder;
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
  std::cout.precision(0);
  std::cout << "<table border=\"1\">\n"
            << "<tr>"
            << "<th>Particles</th> <th>Levels</th> <th>Time (ns)</th>\n";
  for (std::size_t numParticles = 1000; numParticles <= 1000000;
       numParticles *= 10) {
    std::vector<Point> particles(numParticles);

    Float cellLength = 0.0625 * 0.0625;
    const std::size_t MaxLevel = 20;
    for (std::size_t numLevels = 8; numLevels <= MaxLevel;
         numLevels += 4, cellLength *= 0.0625) {
      MortonOrder morton(Domain, cellLength, 0);

      // The particles are uniformly-distributed random points.
      for (std::size_t i = 0; i != particles.size(); ++i) {
        for (std::size_t j = 0; j != Dimension; ++j) {
          particles[i][j] = random();
        }
      }

      morton.setParticles(particles.begin(), particles.end());
      std::random_shuffle(morton.particles.begin(), morton.particles.end());

      timer.tic();
      morton.order();
      elapsedTime = timer.toc();
      result += std::size_t(stlib::ext::sum(particles[0]));

      std::cout << "<tr><td>" << numParticles << "</td><td>"
                << morton.morton.numLevels() << "</td><td>"
                << 1e9 * elapsedTime / numParticles << "</td>\n";
    }
  }
  std::cout << "</table>\n";

  std::cout << "\nMeaningless result = " << result << "\n";

  return 0;
}
