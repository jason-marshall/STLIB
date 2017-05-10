// -*- C++ -*-

#include "stlib/particle/orderMpi.h"
#include "stlib/particle/traits.h"
#include "stlib/ads/functor/Identity.h"
#include "stlib/ext/vector.h"
#include "stlib/numerical/random/uniform/ContinuousUniformGenerator.h"

using namespace stlib;

void
checkSum(const std::size_t local, const std::size_t total)
{
  std::size_t sum;
  MPI_Reduce(&local, &sum, 1, MPI_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
  if (mpi::commRank() == 0) {
    assert(sum == total);
  }
}

void
printSizes(const char* name, const std::size_t size)
{
  std::vector<std::size_t> sizes(mpi::commSize());
  MPI_Gather(&size, 1, MPI_LONG, &sizes[0], 1, MPI_LONG, 0, MPI_COMM_WORLD);
  if (mpi::commRank() == 0) {
    std::cout << name << " = " << sizes[0];
    for (std::size_t i = 1; i != sizes.size(); ++i) {
      std::cout << ", " << sizes[i];
    }
    std::cout << '\n';
  }
}

template<typename _Float, std::size_t _Dimension>
struct SetPosition {
  void
  operator()(std::array<_Float, _Dimension>* particle,
             const std::array<_Float, _Dimension>& point) const
  {
    *particle = point;
  }
};

template<typename _Float, std::size_t _Dimension, bool _Periodic>
void
test(const std::size_t numParticles, const _Float interactionDistance)
{
  typedef std::array<_Float, _Dimension> Point;
  typedef Point Particle;
  typedef particle::Traits<Particle, ads::Identity<Particle>,
          SetPosition<_Float, _Dimension>, _Periodic,
          _Dimension, _Float> Traits;
  typedef particle::MortonOrderMpi<Traits> MortonOrder;
  typedef numerical::ContinuousUniformGeneratorOpen<>
  ContinuousUniformGenerator;

  const std::size_t rank = mpi::commRank();
  const std::size_t totalParticles = mpi::commSize() * numParticles;
  if (rank == 0) {
    std::cout << "_Dimension = " << _Dimension
              << ", numParticles = " << numParticles
              << ", interactionDistance = " << interactionDistance << '\n';
  }

  const geom::BBox<_Float, _Dimension> Domain =
  {ext::filled_array<Point>(0), ext::filled_array<Point>(1)};
  const _Float ShadowWidth = interactionDistance;
  const _Float Padding = 0.1 * interactionDistance;
  MortonOrder mortonOrder(MPI_COMM_WORLD, Domain, interactionDistance,
                          ShadowWidth, Padding);
  assert(mortonOrder.partitionCount() == 0);
  assert(mortonOrder.reorderCount() == 0);

  // The random number generator. Seed with the rank.
  ContinuousUniformGenerator::DiscreteUniformGenerator generator(rank);
  ContinuousUniformGenerator random(&generator);

  // Make a random vector of particles with random positions.
  std::vector<Particle> particles(numParticles);
  for (std::size_t i = 0; i != particles.size(); ++i) {
    for (std::size_t j = 0; j != _Dimension; ++j) {
      particles[i][j] = random();
    }
  }

  // Calculate the morton order.
  mortonOrder.setParticles(particles.begin(), particles.end());
  mortonOrder.checkValidity();
  mortonOrder.exchangeParticles();
  // The partition count should be 2. The first partition uses the number
  // of particles, while the second uses the number of neighbors.
  assert(mortonOrder.partitionCount() == 2);
  assert(mortonOrder.reorderCount() == 0);
  checkSum(mortonOrder.localParticlesEnd() -
           mortonOrder.localParticlesBegin(), totalParticles);
  printSizes("local", mortonOrder.localParticlesEnd() -
             mortonOrder.localParticlesBegin());
  printSizes("all", mortonOrder.particles.size());

  // Move by a small amount.
  for (std::size_t i = 0; i != mortonOrder.particles.size(); ++i) {
    for (std::size_t j = 0; j != _Dimension; ++j) {
      mortonOrder.particles[i][j] += 0.1 * Padding * random();
    }
  }

  if (mortonOrder.repair()) {
    mortonOrder.exchangeParticles();
  }
  mortonOrder.checkValidity();
  assert(mortonOrder.partitionCount() == 2);
  assert(mortonOrder.reorderCount() == 0);
  checkSum(mortonOrder.localParticlesEnd() -
           mortonOrder.localParticlesBegin(), totalParticles);
  printSizes("local", mortonOrder.localParticlesEnd() -
             mortonOrder.localParticlesBegin());
  printSizes("all", mortonOrder.particles.size());

  // Move to new random positions.
  for (std::size_t i = 0; i != mortonOrder.particles.size(); ++i) {
    for (std::size_t j = 0; j != _Dimension; ++j) {
      mortonOrder.particles[i][j] = random();
    }
  }

  if (mortonOrder.repair()) {
    mortonOrder.exchangeParticles();
  }
  mortonOrder.checkValidity();
  assert(mortonOrder.partitionCount() >= 2);
  checkSum(mortonOrder.localParticlesEnd() -
           mortonOrder.localParticlesBegin(), totalParticles);
  printSizes("local", mortonOrder.localParticlesEnd() -
             mortonOrder.localParticlesBegin());
  printSizes("all", mortonOrder.particles.size());
}


int
main(int argc, char* argv[])
{
  MPI_Init(&argc, &argv);
  if (mpi::commRank() == 0) {
    std::cout << "Number of processes = " << mpi::commSize() << '\n';
  }

  for (float interactionDistance = 1. / 8; interactionDistance >= 1. / 4096;
       interactionDistance *= 1. / 8) {
    for (std::size_t i = 1; i <= 100; i *= 10) {
      test<float, 1, false>(i, interactionDistance);
      test<float, 2, false>(i, interactionDistance);
      test<float, 3, false>(i, interactionDistance);
      test<float, 1, true>(i, interactionDistance);
      test<float, 2, true>(i, interactionDistance);
      test<float, 3, true>(i, interactionDistance);
    }
  }

  MPI_Finalize();
  return 0;
}
