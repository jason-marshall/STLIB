// -*- C++ -*-

#include "stlib/particle/order.h"
#include "stlib/particle/traits.h"
#include "stlib/particle/verlet.h"

#include "stlib/ads/functor/Identity.h"
#include "stlib/ads/functor/coordinateCompare.h"
#include "stlib/ads/timer/Timer.h"
#include "stlib/container/PackedArrayOfArrays.h"
#include "stlib/geom/orq/CellArrayNeighbors.h"
#include "stlib/numerical/random/uniform/ContinuousUniformGenerator.h"

#include <iostream>

using namespace stlib;

typedef float Float;
const std::size_t Dimension = 3;
// A particle is just a point.
typedef std::array<Float, Dimension> Point;

template<typename _PackedArray>
inline
Float
sumSquaredDistance(const std::vector<Point>& particles,
                   const _PackedArray& neighbors)
{
  Float sum = 0;
  for (std::size_t i = 0; i != neighbors.numArrays(); ++i) {
    for (std::size_t j = 0; j != neighbors.size(i); ++j) {
      sum += stlib::ext::squaredDistance(particles[i],
                                         particles[neighbors(i, j)]);
    }
  }
  return sum;
}

int
main()
{
  typedef geom::CellArrayNeighbors<Float, Dimension,
          std::vector<Point>::iterator>
          CellArrayNeighbors;
  typedef particle::PlainTraits<Point, ads::Identity<Point>, Dimension, Float>
  Traits;
  typedef particle::MortonOrder<Traits> MortonOrder;
  typedef numerical::ContinuousUniformGeneratorOpen<>
  ContinuousUniformGenerator;

  ContinuousUniformGenerator::DiscreteUniformGenerator generator;
  ContinuousUniformGenerator random(&generator);

  ads::Timer timer;
  double elapsedTime;

  const geom::BBox<Float, Dimension> Domain = {ext::filled_array<Point>(0),
                                               ext::filled_array<Point>(1)
                                              };

  std::cout.setf(std::ios_base::fixed, std::ios_base::floatfield);
  std::cout << "Particles,Unordered,Sorted,Morton\n";
  for (std::size_t numParticles = 1024; numParticles <= 4 * 1024 * 1024;
       numParticles *= 4) {
    std::cout << numParticles;

    // The particles are uniformly-distributed random points.
    std::vector<Point> particles(numParticles);
    for (std::size_t i = 0; i != particles.size(); ++i) {
      for (std::size_t j = 0; j != Dimension; ++j) {
        particles[i][j] = random();
      }
    }

    const Float radius = 2 * std::pow(1. / particles.size(), 1. / Dimension);
    container::PackedArrayOfArrays<std::size_t> neighbors;

    {
      CellArrayNeighbors cellArray;
      typedef CellArrayNeighbors::Record Record;
      cellArray.initialize(particles.begin(), particles.end());
      neighbors.clear();
      std::vector<Record> records;
      for (std::size_t i = 0; i != particles.size(); ++i) {
        neighbors.pushArray();
        records.clear();
        cellArray.neighborQuery(particles[i], radius, &records);
        for (std::size_t j = 0; j != records.size(); ++j) {
          neighbors.push_back(std::distance(particles.begin(),
                                            records[j]));
        }
      }
      timer.tic();
      const Float sum = sumSquaredDistance(particles, neighbors);
      elapsedTime = timer.toc();
      // Use the result in a trivial way.
      elapsedTime += std::numeric_limits<Float>::epsilon() *
                     (sum - std::floor(sum));
      std::cout << ',' << elapsedTime * 1e9 / neighbors.size();
    }
    {
      ads::LessThanCompareCoordinate<Point> compare(2);
      std::sort(particles.begin(), particles.end(), compare);
      CellArrayNeighbors cellArray;
      typedef CellArrayNeighbors::Record Record;
      cellArray.initialize(particles.begin(), particles.end());
      neighbors.clear();
      std::vector<Record> records;
      for (std::size_t i = 0; i != particles.size(); ++i) {
        neighbors.pushArray();
        records.clear();
        cellArray.neighborQuery(particles[i], radius, &records);
        for (std::size_t j = 0; j != records.size(); ++j) {
          neighbors.push_back(std::distance(particles.begin(),
                                            records[j]));
        }
      }
      timer.tic();
      const Float sum = sumSquaredDistance(particles, neighbors);
      elapsedTime = timer.toc();
      // Use the result in a trivial way.
      elapsedTime += std::numeric_limits<Float>::epsilon() *
                     (sum - std::floor(sum));
      std::cout << ',' << elapsedTime * 1e9 / neighbors.size();
    }
    {
      MortonOrder morton(Domain, radius, 0);
      particle::VerletLists<MortonOrder> verlet(morton);
      morton.setParticles(particles.begin(), particles.end());
      verlet.findAllNeighbors();
      neighbors.clear();
      for (std::size_t i = 0; i != morton.particles.size(); ++i) {
        neighbors.pushArray();
        for (std::size_t j = 0; j != verlet.neighbors.size(i); ++j) {
          neighbors.push_back(verlet.neighbors(i, j).particle);
        }
      }
      timer.tic();
      const Float sum = sumSquaredDistance(morton.particles, neighbors);
      elapsedTime = timer.toc();
      // Use the result in a trivial way.
      elapsedTime += std::numeric_limits<Float>::epsilon() *
                     (sum - std::floor(sum));
      std::cout << ',' << elapsedTime * 1e9 / neighbors.size() << '\n';
    }
  }

  return 0;
}
