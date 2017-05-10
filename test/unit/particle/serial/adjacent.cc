// -*- C++ -*-

#include "stlib/particle/adjacent.h"
#include "stlib/particle/order.h"
#include "stlib/particle/traits.h"
#include "stlib/numerical/random/uniform/ContinuousUniformGenerator.h"
#include "stlib/ads/functor/Identity.h"

#include <set>

using namespace stlib;

template<typename _Float, std::size_t _Dimension>
struct SetPosition {
  void
  operator()(std::array<_Float, _Dimension>* particle,
             const std::array<_Float, _Dimension>& point) const
  {
    *particle = point;
  }
};


template<typename _Order, typename _Neighbors>
void
testNeighbors(const _Order& mortonOrder, const _Neighbors& neighbors)
{
  typedef typename _Order::Float Float;
  typedef typename _Order::Point Point;

  const Float actual = mortonOrder.squaredInteractionDistance();
  std::set<std::size_t> neighborSet;
  std::vector<std::size_t> indices;
  std::vector<Point> positions;
  // For each cell.
  for (std::size_t cell = 0; cell != mortonOrder.cellsSize(); ++cell) {
    neighbors.positionsInAdjacent(cell, &indices, &positions);
    // For each particle in the cell.
    for (std::size_t i = mortonOrder.cellBegin(cell);
         i != mortonOrder.cellEnd(cell); ++i) {
      assert(indices.size() == neighbors.numPotentialNeighbors(i));
      assert(positions.size() == neighbors.numPotentialNeighbors(i));
      // For each neighbor.
      for (std::size_t j = 0; j != indices.size(); ++j) {
        const Float d = stlib::ext::squaredDistance(mortonOrder.particles[i],
                                        positions[j]);
        if (neighbors.isNeighbor(i, j)) {
          assert(d <= actual);
        }
        else {
          assert(i == indices[j] || d > actual || d != d);
        }
      }

      // Make a set of the neighbors.
      neighborSet.clear();
      for (std::size_t j = 0; j != indices.size(); ++j) {
        if (neighbors.isNeighbor(i, j)) {
          neighborSet.insert(indices[j]);
        }
      }
      // For every particle that is not a neighbor.
      for (std::size_t j = 0; j != mortonOrder.particles.size(); ++j) {
        if (j == i) {
          // A particle is not its own neighbor.
          assert(neighborSet.count(j) == 0);
        }
        else if (neighborSet.count(j) == 0) {
          assert(stlib::ext::squaredDistance(mortonOrder.particles[i],
                                             mortonOrder.particles[j]) >
                 actual);
        }
      }
    }
  }
}


template<std::size_t _Dimension, bool _Periodic, typename _Float>
void
test(const _Float interactionDistance, const _Float padding)
{
  typedef std::array<_Float, _Dimension> Particle;
  typedef particle::Traits<Particle, ads::Identity<Particle>,
          SetPosition<_Float, _Dimension>, _Periodic,
          _Dimension, _Float> Traits;
  typedef particle::MortonOrder<Traits> MortonOrder;

  typedef typename MortonOrder::Point Point;
  typedef numerical::ContinuousUniformGeneratorOpen<>
  ContinuousUniformGenerator;

  // Make the data structure for ordering the particles.
  const geom::BBox<_Float, _Dimension> Domain = {ext::filled_array<Point>(0),
                                                 ext::filled_array<Point>(1)
                                                };
  MortonOrder mortonOrder(Domain, interactionDistance, padding);
  {
    // The random number generator.
    ContinuousUniformGenerator::DiscreteUniformGenerator generator;
    ContinuousUniformGenerator random(&generator);

    // Make a vector of particles with random positions.
    std::vector<Point> particles(100);
    for (std::size_t i = 0; i != particles.size(); ++i) {
      for (std::size_t j = 0; j != _Dimension; ++j) {
        particles[i][j] = random();
      }
    }

    // Order the particles.
    mortonOrder.setParticles(particles.begin(), particles.end());
  }

  {
    particle::AdjacentMask<MortonOrder> neighbors(mortonOrder);
    neighbors.findLocalNeighbors();
    testNeighbors(mortonOrder, neighbors);
  }

  {
    particle::AdjacentMask<MortonOrder> neighbors(mortonOrder);
    neighbors.findAllNeighbors();
    testNeighbors(mortonOrder, neighbors);
  }
}


int
main()
{
  {
    float length = 1;
    for (std::size_t i = 0; i != 5; ++i, length *= 0.5) {
      test<1, false>(length, float(0));
      test<2, false>(length, float(0));
      test<3, false>(length, float(0));
      test<1, true>(float(0.25) * length, float(0));
      test<2, true>(float(0.25) * length, float(0));
      test<3, true>(float(0.25) * length, float(0));

      test<1, false>(float(0.5) * length, float(0.5) * length);
      test<2, false>(float(0.5) * length, float(0.5) * length);
      test<3, false>(float(0.5) * length, float(0.5) * length);
      test<1, true>(float(0.125) * length, float(0.125) * length);
      test<2, true>(float(0.125) * length, float(0.125) * length);
      test<3, true>(float(0.125) * length, float(0.125) * length);
    }
  }
  {
    double length = 1;
    for (std::size_t i = 0; i != 5; ++i, length *= 0.5) {
      test<1, false>(length, double(0));
      test<2, false>(length, double(0));
      test<3, false>(length, double(0));
      test<1, true>(0.25 * length, double(0));
      test<2, true>(0.25 * length, double(0));
      test<3, true>(0.25 * length, double(0));

      test<1, false>(0.5 * length, 0.5 * length);
      test<2, false>(0.5 * length, 0.5 * length);
      test<3, false>(0.5 * length, 0.5 * length);
      test<1, true>(0.125 * length, 0.125 * length);
      test<2, true>(0.125 * length, 0.125 * length);
      test<3, true>(0.125 * length, 0.125 * length);
    }
  }

  return 0;
}
