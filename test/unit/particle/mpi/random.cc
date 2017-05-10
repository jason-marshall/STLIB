// -*- C++ -*-

#include "stlib/particle/orderMpi.h"
#include "stlib/particle/traits.h"
#include "stlib/ads/utility/ParseOptionsArguments.h"
#include "stlib/numerical/random/uniform/ContinuousUniformGenerator.h"

USING_STLIB_EXT_ARRAY_MATH_OPERATORS;
using namespace stlib;

template<typename _Float, std::size_t _Dimension>
struct Particle {
  std::size_t identifier;
  std::array<_Float, _Dimension> position;
};

template<typename _Float, std::size_t _Dimension>
struct GetPosition :
    public std::unary_function<Particle<_Float, _Dimension>,
    std::array<_Float, _Dimension> > {
  typedef std::unary_function<Particle<_Float, _Dimension>,
          std::array<_Float, _Dimension> > Base;
  const typename Base::result_type&
  operator()(const typename Base::argument_type& x) const
  {
    return x.position;
  }
};

template<typename _Float, std::size_t _Dimension>
struct SetPosition {
  void
  operator()(Particle<_Float, _Dimension>* particle,
             const std::array<_Float, _Dimension>& point) const
  {
    particle->position = point;
  }
};

void
printSizes(const std::size_t size)
{
  std::vector<std::size_t> sizes(mpi::commSize());
  MPI_Gather(&size, 1, MPI_LONG, &sizes[0], 1, MPI_LONG, 0, MPI_COMM_WORLD);
  if (mpi::commRank() == 0) {
    std::cout << "sizes = " << sizes[0];
    for (std::size_t i = 1; i != sizes.size(); ++i) {
      std::cout << ", " << sizes[i];
    }
    std::cout << '\n';
  }
}

template<typename _Float, std::size_t _Dimension, bool _Periodic>
void
test(const _Float interactionDistance, const _Float shadowWidth,
     const _Float padding,
     const _Float velocity, const std::size_t numParticles,
     const std::size_t numSteps)
{
  typedef std::array<_Float, _Dimension> Point;
  typedef Particle<_Float, _Dimension> Particle;
  typedef particle::Traits<Particle, GetPosition<_Float, _Dimension>,
          SetPosition<_Float, _Dimension>, _Periodic,
          _Dimension, _Float> Traits;
  typedef particle::MortonOrderMpi<Traits> MortonOrder;
  typedef numerical::ContinuousUniformGeneratorOpen<>
  ContinuousUniformGenerator;

  const std::size_t commRank = mpi::commRank();
  const std::size_t commSize = mpi::commSize();
  if (commRank == 0) {
    std::cout << "_Dimension = " << _Dimension
              << ", numParticles = " << numParticles << '\n';
  }

  const geom::BBox<_Float, _Dimension> Domain =
  {ext::filled_array<Point>(0), ext::filled_array<Point>(1)};
  MortonOrder mortonOrder(MPI_COMM_WORLD, Domain, interactionDistance,
                          shadowWidth, padding);

  // The random number generator. Seed with the rank.
  ContinuousUniformGenerator::DiscreteUniformGenerator generator(commRank);
  ContinuousUniformGenerator random(&generator);

  // Make a random vector of particles with random positions.
  std::vector<Particle> particles(numParticles);
  for (std::size_t i = 0; i != particles.size(); ++i) {
    particles[i].identifier = commRank * numParticles + i;
    for (std::size_t j = 0; j != _Dimension; ++j) {
      particles[i].position[j] = random();
    }
  }

  // Calculate the morton order.
  mortonOrder.setParticles(particles.begin(), particles.end());
  mortonOrder.checkValidity();
  printSizes(mortonOrder.particles.size());

  Point v;
  for (std::size_t n = 0; n != numSteps; ++n) {
    mortonOrder.repair();
    mortonOrder.exchangeParticles();
    for (std::size_t i = 0; i != mortonOrder.particles.size(); ++i) {
      // Determine a displacement vector.
      for (std::size_t j = 0; j != _Dimension; ++j) {
        v[j] = 2 * random() - 1;
      }
      stlib::ext::normalize(&v);
      v *= velocity;
      // Move the particle.
      mortonOrder.particles[i].position += v;
      // Keep the particle inside the unit box.
      for (std::size_t j = 0; j != _Dimension; ++j) {
        _Float& x = mortonOrder.particles[i].position[j];
        if (x < 0) {
          x = 0;
        }
        if (x > 1) {
          x = 1;
        }
      }
    }
  }
  mortonOrder.checkValidity();
  printSizes(mortonOrder.particles.size());

  // Check the sum of the identifiers.
  std::size_t sumIds = 0;
  for (std::size_t i = mortonOrder.localParticlesBegin();
       i != mortonOrder.localParticlesEnd(); ++i) {
    sumIds += mortonOrder.particles[i].identifier;
  }
  std::size_t globalSum;
  MPI_Reduce(&sumIds, &globalSum, 1, MPI_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
  if (commRank == 0) {
    const std::size_t n = commSize * numParticles;
    assert(globalSum == n * (n - 1) / 2);
  }
}


int
main(int argc, char* argv[])
{
  MPI_Init(&argc, &argv);

  ads::ParseOptionsArguments parser(argc, argv);

  float interactionDistance = 0.1;
  parser.getOption('i', &interactionDistance);
  assert(interactionDistance > 0);

  float shadowWidth = interactionDistance;
  parser.getOption('w', &shadowWidth);
  assert(shadowWidth >= interactionDistance);

  float padding = 0.1 * interactionDistance;
  parser.getOption('p', &padding);
  assert(padding > 0);

  float velocity = 0.01 * interactionDistance;
  parser.getOption('v', &velocity);
  assert(velocity >= 0);

  std::size_t numParticlesPerProcess = 10;
  parser.getOption('n', &numParticlesPerProcess);
  assert(numParticlesPerProcess > 0);

  std::size_t numSteps = 10;
  parser.getOption('s', &numSteps);
  assert(numSteps > 0);

  test<float, 1, false>(interactionDistance, shadowWidth, padding, velocity,
                        numParticlesPerProcess, numSteps);
  test<float, 2, false>(interactionDistance, shadowWidth, padding, velocity,
                        numParticlesPerProcess, numSteps);
  test<float, 3, false>(interactionDistance, shadowWidth, padding, velocity,
                        numParticlesPerProcess, numSteps);
  test<float, 1, true>(interactionDistance, shadowWidth, padding, velocity,
                       numParticlesPerProcess, numSteps);
  test<float, 2, true>(interactionDistance, shadowWidth, padding, velocity,
                       numParticlesPerProcess, numSteps);
  test<float, 3, true>(interactionDistance, shadowWidth, padding, velocity,
                       numParticlesPerProcess, numSteps);

  MPI_Finalize();
  return 0;
}
